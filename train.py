# train.py

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import glob
import argparse
from time import time

# --- NEW: Import torch.multiprocessing ---
import torch.multiprocessing

# --- W&B INTEGRATION: Import the library ---
import wandb

# MONAI imports
from monai.networks.nets import SwinUNETR
from monai.losses import DiceCELoss

# Import from our new transforms file
from transforms import get_train_transforms

class BraTS2D5Dataset(Dataset):
    def __init__(self, data_dir, image_size, spacing, num_patients=None):
        self.image_size = image_size
        patient_dirs = sorted(glob.glob(os.path.join(data_dir, "BraTS*")))
        
        if num_patients is not None:
            print(f"--- Using a subset of {num_patients} patients for testing. ---")
            patient_dirs = patient_dirs[:num_patients]
        
        if not patient_dirs:
            raise FileNotFoundError(f"No patient data found in '{data_dir}'. Check your --data_dir path.")
        
        self.files = []
        for patient_dir in patient_dirs:
            self.files.append({
                "t1": glob.glob(os.path.join(patient_dir, "*-t1n.nii.gz"))[0],
                "t1ce": glob.glob(os.path.join(patient_dir, "*-t1c.nii.gz"))[0],
                "t2": glob.glob(os.path.join(patient_dir, "*-t2w.nii.gz"))[0],
                "flair": glob.glob(os.path.join(patient_dir, "*-t2f.nii.gz"))[0],
                "label": glob.glob(os.path.join(patient_dir, "*-seg.nii.gz"))[0],
            })
        
        transforms = get_train_transforms(image_size, spacing)

        # --- OPTIMIZATION: Pre-load and process all volumes into memory ---
        print("--- Pre-loading and processing volumes... ---")
        start_time = time()
        self.processed_volumes = []
        for i, patient_files in enumerate(self.files):
            self.processed_volumes.append(transforms(patient_files))
            if (i + 1) % 10 == 0 or (i + 1) == len(self.files):
                print(f"  Processed {i + 1}/{len(self.files)} patients...")
        print(f"--- Volume processing took {time() - start_time:.2f} seconds. ---")


        self.slice_map = []
        print("Mapping slices to volumes...")
        for vol_idx, p_data in enumerate(self.processed_volumes):
            num_slices = p_data["label"].shape[3]
            for slice_idx in range(num_slices):
                self.slice_map.append((vol_idx, slice_idx))
                
        print(f"Dataset ready. Found {len(self.slice_map)} total slices from {len(self.files)} volumes.")


    def __len__(self):
        return len(self.slice_map)

    def __getitem__(self, index):
        # --- OPTIMIZATION: Directly access pre-processed data ---
        volume_idx, slice_idx = self.slice_map[index]
        patient_data = self.processed_volumes[volume_idx]
        
        img_modalities = torch.cat([patient_data['t1'], patient_data['t1ce'], patient_data['t2'], patient_data['flair']], dim=0)
        label_volume = patient_data['label']

        # Core 2.5D Logic
        num_slices_in_vol = img_modalities.shape[3]
        prev_slice_idx = max(0, slice_idx - 1)
        next_slice_idx = min(num_slices_in_vol - 1, slice_idx + 1)
        
        stacked_slices = torch.stack([
            img_modalities[:, :, :, prev_slice_idx], 
            img_modalities[:, :, :, slice_idx], 
            img_modalities[:, :, :, next_slice_idx]
        ], dim=0)

        in_channels = 4 * 3
        input_tensor = stacked_slices.view(in_channels, self.image_size[0], self.image_size[1])
        target_tensor = label_volume[:, :, :, slice_idx]

        return input_tensor, target_tensor

def get_args():
    # ... (This function is unchanged) ...
    parser = argparse.ArgumentParser(description="2.5D Swin UNETR training for BraTS.")
    parser.add_argument('--data_dir', type=str, required=True, help='Root directory for the BraTS dataset.')
    parser.add_argument('--output_dir', type=str, default='./checkpoints', help='Directory to save model checkpoints.')
    parser.add_argument('--epochs', type=int, default=25, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=4, help='Training batch size.')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate.')
    parser.add_argument('--img_size', type=int, default=128, help='Image size (height and width).')
    parser.add_argument(
        '--num_patients',
        type=int,
        default=None,
        help='Number of patient volumes to use for quick testing (default: all).'
    )
    return parser.parse_args()

def main(args):
    # --- NEW: Set the multiprocessing sharing strategy to prevent deadlocks ---
    torch.multiprocessing.set_sharing_strategy('file_system')

    # --- W&B INTEGRATION: Step 1 - Initialize the run ---
    run_name = f"swin_unetr_2.5d_{int(time())}"
    wandb.init(config=args, name=run_name)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    dataset = BraTS2D5Dataset(
        data_dir=args.data_dir,
        image_size=(args.img_size, args.img_size),
        spacing=(1.0, 1.0, 1.0),
        num_patients=args.num_patients
    )
    # --- NOTE: Using num_workers is now safe and efficient ---
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=6)

    model = SwinUNETR(
        in_channels=12,
        out_channels=4, 
        feature_size=24,
        spatial_dims=2,
    ).to(device)

    wandb.watch(model, log="all", log_freq=100)

    loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    print("Starting training...")
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        num_batches = len(data_loader)
        
        for i, (inputs, labels) in enumerate(data_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
            # --- NEW: Log metrics and image sample every 10 batches ---
            if (i + 1) % 10 == 0:
                print(f"  Epoch {epoch + 1}/{args.epochs}, Batch {i + 1}/{num_batches}...")
                
                # Get the first image in the batch for visualization
                # The 5th channel (index 4) is the T1c modality of the center slice
                input_image = inputs[0, 4, :, :].cpu().numpy()
                
                # Prepare masks for wandb.Image
                # Class labels for wandb are defined here
                class_labels = {1: "necrotic core", 2: "edema", 3: "enhancing tumor"}
                
                # Get ground truth mask from the label tensor
                gt_mask = labels[0, 0, :, :].cpu().numpy()
                
                # Get prediction mask by taking argmax of model output
                pred_mask = torch.argmax(outputs, dim=1)[0].cpu().numpy()
                
                # Log to wandb
                wandb.log({
                    "batch_loss": loss.item(),
                    "segmentation_sample": wandb.Image(
                        input_image,
                        masks={
                            "ground_truth": {"mask_data": gt_mask, "class_labels": class_labels},
                            "prediction": {"mask_data": pred_mask, "class_labels": class_labels},
                        },
                    ),
                })
        
        avg_loss = epoch_loss / num_batches
        print(f"--- Epoch {epoch + 1}/{args.epochs}, Average Loss: {avg_loss:.4f} ---")
        
        # Log epoch-level metrics
        wandb.log({"epoch": epoch + 1, "avg_loss": avg_loss})
        
        checkpoint_path = os.path.join(args.output_dir, f"swin_unetr_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

    print("Training finished!")
    wandb.finish()


if __name__ == '__main__':
    args = get_args()
    main(args)