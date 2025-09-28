import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import glob
import argparse

# MONAI imports
from monai.networks.nets import SwinUNETR
from monai.losses import DiceCELoss
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    ScaleIntensityRanged,
    CropForegroundd,
    Resized,
)

# --- 1. Custom Dataset for Real BraTS Data ---
# This class is designed to load real NIfTI files from a directory.
class BraTS2D5Dataset(Dataset):
    def __init__(self, data_dir, image_size, spacing):
        self.image_size = image_size
        self.patient_dirs = sorted(glob.glob(os.path.join(data_dir, "BraTS*")))
        if not self.patient_dirs:
            raise FileNotFoundError(f"No patient data found in '{data_dir}'. Check your --data_dir path.")
        
        self.files = []
        for patient_dir in self.patient_dirs:
            # Assumes standard BraTS file naming
            self.files.append({
                "t1": glob.glob(os.path.join(patient_dir, "*-t1n.nii.gz"))[0],
                "t1ce": glob.glob(os.path.join(patient_dir, "*-t1c.nii.gz"))[0],
                "t2": glob.glob(os.path.join(patient_dir, "*-t2w.nii.gz"))[0],
                "flair": glob.glob(os.path.join(patient_dir, "*-t2f.nii.gz"))[0],
                "label": glob.glob(os.path.join(patient_dir, "*-seg.nii.gz"))[0],
            })
        
        # Preprocessing transforms for a single 3D volume
        self.transforms = Compose([
            LoadImaged(keys=["t1", "t1ce", "t2", "flair", "label"]),
            EnsureChannelFirstd(keys=["t1", "t1ce", "t2", "flair", "label"]),
            Orientationd(keys=["t1", "t1ce", "t2", "flair", "label"], axcodes="RAS"),
            Spacingd(keys=["t1", "t1ce", "t2", "flair", "label"], pixdim=spacing, mode=("bilinear", "bilinear", "bilinear", "bilinear", "nearest")),
            ScaleIntensityRanged(keys=["t1", "t1ce", "t2", "flair"], a_min=0, a_max=4000, b_min=0.0, b_max=1.0, clip=True),
            CropForegroundd(keys=["t1", "t1ce", "t2", "flair", "label"], source_key="t1"),
            Resized(keys=["t1", "t1ce", "t2", "flair", "label"], spatial_size=(image_size[0], image_size[1], -1)),
        ])

        # Map a global slice index to a (volume_index, slice_in_volume_index) pair
        self.slice_map = []
        print("Mapping slices to volumes...")
        for vol_idx, patient_files in enumerate(self.files):
            # --- CORRECTED CODE ---
            # To get the slice count, we must load and transform the full data dictionary
            # because transforms like CropForegroundd depend on other keys (e.g., source_key="t1").
            sample_data = self.transforms(patient_files)
            # --- END CORRECTION ---
            
            num_slices = sample_data["label"].shape[3]
            for slice_idx in range(num_slices):
                self.slice_map.append((vol_idx, slice_idx))
        print(f"Dataset ready. Found {len(self.slice_map)} total slices from {len(self.files)} volumes.")

    def __len__(self):
        return len(self.slice_map)

    def __getitem__(self, index):
        volume_idx, slice_idx = self.slice_map[index]
        patient_data = self.transforms(self.files[volume_idx])
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
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="2.5D Swin UNETR training for BraTS.")
    parser.add_argument('--data_dir', type=str, required=True, help='Root directory for the BraTS dataset.')
    parser.add_argument('--output_dir', type=str, default='./checkpoints', help='Directory to save model checkpoints.')
    parser.add_argument('--epochs', type=int, default=25, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=4, help='Training batch size.')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate.')
    parser.add_argument('--img_size', type=int, default=128, help='Image size (height and width).')
    return parser.parse_args()

def main(args):
    """Main training function."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    dataset = BraTS2D5Dataset(
        data_dir=args.data_dir,
        image_size=(args.img_size, args.img_size),
        spacing=(1.0, 1.0, 1.0)
    )
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    model = SwinUNETR(
        img_size=(args.img_size, args.img_size),
        in_channels=12,
        out_channels=4,
        feature_size=24,
        spatial_dims=2,
    ).to(device)

    loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    print("Starting training...")
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        for i, (inputs, labels) in enumerate(data_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(data_loader)
        print(f"--- Epoch {epoch + 1}/{args.epochs}, Average Loss: {avg_loss:.4f} ---")
        
        # Save model checkpoint
        checkpoint_path = os.path.join(args.output_dir, f"swin_unetr_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

    print("Training finished!")

if __name__ == '__main__':
    args = get_args()
    main(args)