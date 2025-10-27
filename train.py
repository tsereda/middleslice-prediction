# train.py

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
    def __init__(self, data_dir, image_size, spacing, num_patients=None, debug=False):
        self.image_size = image_size
        self.debug = debug
        
        patient_dirs = sorted(glob.glob(os.path.join(data_dir, "BraTS*")))
        if num_patients is not None:
            print(f"--- Using a subset of {num_patients} patients for testing. ---")
            patient_dirs = patient_dirs[:num_patients]
        if not patient_dirs:
            raise FileNotFoundError(f"No patient data found in '{data_dir}'. Check your --data_dir path.")
            
        # Fixed file patterns to match BraTS2020 naming convention
        self.files = []
        for p in patient_dirs:
            try:
                file_dict = {
                    "t1": glob.glob(os.path.join(p, "*_t1.nii"))[0],
                    "t1ce": glob.glob(os.path.join(p, "*_t1ce.nii"))[0],
                    "t2": glob.glob(os.path.join(p, "*_t2.nii"))[0],
                    "flair": glob.glob(os.path.join(p, "*_flair.nii"))[0],
                    "label": glob.glob(os.path.join(p, "*_seg.nii"))[0]
                }
                self.files.append(file_dict)
            except IndexError:
                print(f"Warning: Could not find all required files in {p}, skipping...")
                continue
        
        if not self.files:
            raise FileNotFoundError(f"No valid patient data found in '{data_dir}'")
            
        transforms = get_train_transforms(image_size, spacing)
        print("--- Pre-loading and processing volumes... ---")
        start_time = time()
        self.processed_volumes = []
        for i, patient_files in enumerate(self.files):
            self.processed_volumes.append(transforms(patient_files))
            if (i + 1) % 10 == 0 or (i + 1) == len(self.files):
                print(f"   Processed {i + 1}/{len(self.files)} patients...")
        print(f"--- Volume processing took {time() - start_time:.2f} seconds. ---")
        
        # Dynamic threshold detection
        print("--- Analyzing data to determine optimal threshold... ---")
        self.threshold = self._determine_threshold()
        print(f"--- Using threshold: {self.threshold:.6f} ---")
        
        self.slice_map = []
        print("Mapping and filtering slices to create dataset...")
        valid_count = 0
        total_count = 0
        
        for vol_idx, p_data in enumerate(self.processed_volumes):
            num_slices = p_data["label"].shape[3]
            # Ensure we have a valid context (prev and next slice)
            for slice_idx in range(1, num_slices - 1): 
                total_count += 1
                brain_slice = p_data["t1ce"][0, :, :, slice_idx]
                slice_mean = torch.mean(brain_slice).item()
                
                if self.debug and vol_idx == 0:  # Debug first volume
                    print(f"   Volume {vol_idx}, Slice {slice_idx}: mean={slice_mean:.6f}")
                
                if slice_mean > self.threshold:
                    self.slice_map.append((vol_idx, slice_idx))
                    valid_count += 1
        
        print(f"Dataset ready. Found {valid_count} valid slices out of {total_count} total slices from {len(self.files)} volumes.")
        print(f"Valid slice percentage: {(valid_count/total_count)*100:.1f}%")
        
        if len(self.slice_map) == 0:
            print("\nERROR: No valid slices found!")
            print("This usually means:")
            print("1. The threshold is too high for your normalized data")
            print("2. The data preprocessing is normalizing values to very small ranges")
            print("3. There might be an issue with the data loading")
            print(f"\nCurrent threshold: {self.threshold:.6f}")
            print("Consider running debug_data.py to analyze your data distribution")
            raise ValueError("No valid slices found in dataset")

    def _determine_threshold(self):
        """
        Automatically determine a good threshold based on data distribution
        """
        all_means = []
        
        # Sample a few slices from each volume to get distribution
        for vol_idx, p_data in enumerate(self.processed_volumes):
            num_slices = p_data["label"].shape[3]
            # Sample every 5th slice to get a representative sample
            for slice_idx in range(1, num_slices - 1, 5):
                brain_slice = p_data["t1ce"][0, :, :, slice_idx]
                slice_mean = torch.mean(brain_slice).item()
                all_means.append(slice_mean)
        
        all_means = np.array(all_means)
        
        # Print some statistics
        print(f"   Sampled {len(all_means)} slices for threshold analysis")
        print(f"   Mean range: {all_means.min():.6f} to {all_means.max():.6f}")
        print(f"   Overall mean: {all_means.mean():.6f}")
        print(f"   Standard deviation: {all_means.std():.6f}")
        
        # Filter out very low values (likely empty slices)
        non_trivial_means = all_means[all_means > 0.0001]
        
        if len(non_trivial_means) == 0:
            print("   Warning: All sampled slices have very low means, using minimal threshold")
            return 0.0001
        
        # Use 10th percentile of non-trivial values as threshold
        # This should exclude empty slices while keeping most brain tissue
        threshold = np.percentile(non_trivial_means, 10)
        
        # Make sure threshold is not too high
        threshold = min(threshold, 0.01)  # Cap at 0.01 to be safe
        
        return threshold

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
    parser = argparse.ArgumentParser(description="2.5D Swin UNETR for Slice Reconstruction.")
    parser.add_argument('--data_dir', type=str, required=True, help='Root directory for the BraTS dataset.')
    parser.add_argument('--output_dir', type=str, default='./checkpoints', help='Directory to save model checkpoints.')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=8, help='Training batch size.')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate.')
    parser.add_argument('--img_size', type=int, default=256, help='Image size (height and width).')
    parser.add_argument('--num_patients', type=int, default=None, help='Number of patient volumes to use for quick testing (default: all).')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode with additional logging.')
    return parser.parse_args()


def main(args):
    torch.multiprocessing.set_sharing_strategy('file_system')
    run_name = f"swin_unetr_reconstruction_{int(time())}"
    wandb.init(project="brats-2.5d-reconstruction", config=args, name=run_name)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    # Create dataset with debug option
    dataset = BraTS2D5Dataset(
        data_dir=args.data_dir, 
        image_size=(args.img_size, args.img_size), 
        spacing=(1.0, 1.0, 1.0), 
        num_patients=args.num_patients,
        debug=args.debug
    )
    
    # Only create DataLoader if we have valid data
    if len(dataset) == 0:
        print("ERROR: Dataset is empty. Cannot proceed with training.")
        wandb.finish()
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
                
                # The call to the function remains exactly the same
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