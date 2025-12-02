# train.py - Updated for BraTS 2023 with correct modality names

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import glob
import argparse
from time import time
import torch.multiprocessing
import cv2
import wandb
from monai.networks.nets import SwinUNETR
from torch.nn import L1Loss 
from transforms import get_train_transforms
from logging_utils import create_reconstruction_log_panel


class BraTS2D5Dataset(Dataset):
    def __init__(self, data_dir, image_size, spacing, num_patients=None):
        self.image_size = image_size
        patient_dirs = sorted(glob.glob(os.path.join(data_dir, "BraTS*")))
        if num_patients is not None:
            print(f"--- Using a subset of {num_patients} patients for testing. ---")
            patient_dirs = patient_dirs[:num_patients]
        if not patient_dirs:
            raise FileNotFoundError(f"No patient data found in '{data_dir}'. Check your --data_dir path.")
        
        print(f"Found {len(patient_dirs)} patient directories")
        
        # Updated file patterns to match BraTS2023 naming convention
        # BraTS 2023 uses: t1n, t1c, t2w, t2f instead of t1, t1ce, t2, flair
        self.files = []
        skipped_patients = 0
        
        for p in patient_dirs:
            # BraTS 2023 modality patterns with hyphen separators and .nii.gz extension
            t1_patterns = ["*-t1n.nii*", "*_t1n.nii*"]  # T1 native
            t1ce_patterns = ["*-t1c.nii*", "*_t1c.nii*"]  # T1 contrast-enhanced
            t2_patterns = ["*-t2w.nii*", "*_t2w.nii*"]  # T2-weighted
            flair_patterns = ["*-t2f.nii*", "*_t2f.nii*"]  # T2-FLAIR
            seg_patterns = ["*-seg.nii*", "*_seg.nii*"]  # Segmentation
            
            # Find files using multiple patterns
            t1_files = []
            for pattern in t1_patterns:
                t1_files.extend(glob.glob(os.path.join(p, pattern)))
                if t1_files:
                    break
                    
            t1ce_files = []
            for pattern in t1ce_patterns:
                t1ce_files.extend(glob.glob(os.path.join(p, pattern)))
                if t1ce_files:
                    break
                    
            t2_files = []
            for pattern in t2_patterns:
                t2_files.extend(glob.glob(os.path.join(p, pattern)))
                if t2_files:
                    break
                    
            flair_files = []
            for pattern in flair_patterns:
                flair_files.extend(glob.glob(os.path.join(p, pattern)))
                if flair_files:
                    break
                    
            seg_files = []
            for pattern in seg_patterns:
                seg_files.extend(glob.glob(os.path.join(p, pattern)))
                if seg_files:
                    break
            
            # Check if all required files are found
            if not all([t1_files, t1ce_files, t2_files, flair_files, seg_files]):
                print(f"Warning: Missing files in {os.path.basename(p)}")
                print(f"  T1N: {len(t1_files)}, T1C: {len(t1ce_files)}, T2W: {len(t2_files)}, T2F: {len(flair_files)}, SEG: {len(seg_files)}")
                
                # Debug: show what files are actually present
                all_nii_files = glob.glob(os.path.join(p, "*.nii*"))
                print(f"  All .nii* files found: {[os.path.basename(f) for f in all_nii_files]}")
                
                skipped_patients += 1
                continue
                
            self.files.append({
                "t1": t1_files[0],      # T1 native -> t1
                "t1ce": t1ce_files[0],  # T1 contrast -> t1ce
                "t2": t2_files[0],      # T2 weighted -> t2  
                "flair": flair_files[0], # T2 FLAIR -> flair
                "label": seg_files[0]   # Segmentation -> label
            })
        
        if not self.files:
            raise ValueError(f"No valid patient data found. All {len(patient_dirs)} directories were missing required files.")
            
        if skipped_patients > 0:
            print(f"Note: Skipped {skipped_patients} patients due to missing files. Using {len(self.files)} patients.")
        
        # Load and process transforms
        transforms = get_train_transforms(image_size, spacing)
        print("--- Pre-loading and processing volumes... ---")
        start_time = time()
        self.processed_volumes = []
        failed_volumes = 0
        
        for i, patient_files in enumerate(self.files):
            try:
                processed_volume = transforms(patient_files)
                self.processed_volumes.append(processed_volume)
                if (i + 1) % 10 == 0 or (i + 1) == len(self.files):
                    print(f"   Processed {i + 1}/{len(self.files)} patients...")
            except Exception as e:
                print(f"   Error processing patient {i+1}: {e}")
                failed_volumes += 1
                continue
                
        if failed_volumes > 0:
            print(f"Warning: Failed to process {failed_volumes} volumes. Using {len(self.processed_volumes)} volumes.")
            
        if not self.processed_volumes:
            raise ValueError("No volumes could be processed successfully!")
            
        print(f"--- Volume processing took {time() - start_time:.2f} seconds. ---")
        
        # Create slice mapping
        self.slice_map = []
        print("Mapping all available slices to create dataset...")
        for vol_idx, p_data in enumerate(self.processed_volumes):
            num_slices = p_data["label"].shape[3]
            # Ensure we have a valid context (prev and next slice)
            for slice_idx in range(1, num_slices - 1): 
                self.slice_map.append((vol_idx, slice_idx))
        print(f"Dataset ready. Using all {len(self.slice_map)} available slices from {len(self.processed_volumes)} volumes.")

    def __len__(self):
        return len(self.slice_map)
    
    def __getitem__(self, index):
        volume_idx, slice_idx = self.slice_map[index]
        patient_data = self.processed_volumes[volume_idx]
        
        img_modalities = torch.cat([patient_data['t1'], patient_data['t1ce'], patient_data['t2'], patient_data['flair']], dim=0)
        
        prev_slice = img_modalities[:, :, :, slice_idx - 1]
        next_slice = img_modalities[:, :, :, slice_idx + 1]
        input_tensor = torch.cat([prev_slice, next_slice], dim=0)

        target_tensor = img_modalities[:, :, :, slice_idx]

        return input_tensor, target_tensor, slice_idx


def get_args():
    parser = argparse.ArgumentParser(description="2.5D Swin UNETR for Slice Reconstruction - BraTS 2023 Compatible.")
    parser.add_argument('--data_dir', type=str, required=True, help='Root directory for the BraTS dataset.')
    parser.add_argument('--output_dir', type=str, default='./checkpoints', help='Directory to save model checkpoints.')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=8, help='Training batch size.')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate.')
    parser.add_argument('--img_size', type=int, default=256, help='Image size (height and width).')
    parser.add_argument('--num_patients',type=int,default=None,help='Number of patient volumes to use for quick testing (default: all).')
    return parser.parse_args()


def main(args):
    torch.multiprocessing.set_sharing_strategy('file_system')
    run_name = f"swin_unetr_reconstruction_brats2023_{int(time())}"
    wandb.init(project="brats2023-2.5d-reconstruction", config=args, name=run_name, entity="")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    # Create dataset with enhanced error handling
    try:
        dataset = BraTS2D5Dataset(
            data_dir=args.data_dir, 
            image_size=(args.img_size, args.img_size), 
            spacing=(1.0, 1.0, 1.0), 
            num_patients=args.num_patients
        )
    except Exception as e:
        print(f"Error creating dataset: {e}")
        print("Please check your data directory and file structure.")
        return
        
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)

    model = SwinUNETR(in_channels=8, out_channels=4, feature_size=24, spatial_dims=2).to(device)
    wandb.watch(model, log="all", log_freq=100)
    
    loss_function = L1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    print("Starting training for slice reconstruction...")
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        num_batches = len(data_loader)
        
        for i, (inputs, targets, slice_indices) in enumerate(data_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
            if (i + 1) % 100 == 0:
                print(f"   Epoch {epoch + 1}/{args.epochs}, Batch {i + 1}/{num_batches} | L1 Loss: {loss.item():.4f}")
                
                # Log reconstruction samples
                panel_bgr = create_reconstruction_log_panel(
                    inputs[0].detach(), 
                    targets[0].detach(), 
                    outputs[0].detach(), 
                    slice_indices[0].item(), 
                    i + 1
                )
                
                panel_rgb = cv2.cvtColor(panel_bgr, cv2.COLOR_BGR2RGB)
                
                wandb.log({
                    "batch_l1_loss": loss.item(),
                    "reconstruction_samples": wandb.Image(panel_rgb)
                })
        
        avg_loss = epoch_loss / num_batches
        print(f"--- Epoch {epoch + 1}/{args.epochs}, Average L1 Loss: {avg_loss:.4f} ---")
        wandb.log({"epoch": epoch + 1, "avg_epoch_l1_loss": avg_loss})
        
        checkpoint_path = os.path.join(args.output_dir, f"swin_unetr_recon_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

    print("Training finished!")
    wandb.finish()


if __name__ == '__main__':
    args = get_args()
    main(args)