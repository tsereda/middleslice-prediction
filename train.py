import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import glob
import argparse
from time import time
import torch.multiprocessing
import wandb
from monai.networks.nets import SwinUNETR
from torch.nn import L1Loss 
from transforms import get_train_transforms
import cv2

# --- REFINED VISUALIZATION for RECONSTRUCTION (4x4 Panel) ---
def create_reconstruction_log_panel(
    inputs_sample,      # Model input (Prev/Next slices), shape [8, H, W]
    target_sample,      # Ground Truth (Real middle slice), shape [4, H, W]
    output_sample,      # Model Prediction (Reconstructed middle slice), shape [4, H, W]
    slice_idx,
    batch_idx
):
    """Creates a single, 4x4 composite grid for the reconstruction task."""
    
    modalities = ["t1", "t1ce", "t2", "flair"]
    all_rows = []
    header_height = 30
    
    # --- Create a row for each modality ---
    for i, name in enumerate(modalities):
        # Inputs to the model
        prev_slice = (inputs_sample[i].cpu().numpy() * 255).astype(np.uint8)
        next_slice = (inputs_sample[i + 4].cpu().numpy() * 255).astype(np.uint8)
        
        # Ground Truth and Model Prediction
        gt_middle = (target_sample[i].cpu().numpy() * 255).astype(np.uint8)
        pred_middle = (output_sample[i].cpu().numpy() * 255).astype(np.uint8)

        # Convert all to BGR for display
        prev_bgr = cv2.cvtColor(prev_slice, cv2.COLOR_GRAY2BGR)
        next_bgr = cv2.cvtColor(next_slice, cv2.COLOR_GRAY2BGR)
        gt_bgr = cv2.cvtColor(gt_middle, cv2.COLOR_GRAY2BGR)
        pred_bgr = cv2.cvtColor(pred_middle, cv2.COLOR_GRAY2BGR)

        # Combine into a 1x4 strip
        row = np.hstack([prev_bgr, next_bgr, pred_bgr, gt_bgr])
        
        # Create a clean header with all text, no image overlays
        header = np.full((header_height, row.shape[1], 3), 40, dtype=np.uint8)
        cv2.putText(header, f"{name.upper()}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        col_width = prev_bgr.shape[1]
        cv2.putText(header, f"Input (Z-1)", (col_width*0)+10, 20, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
        cv2.putText(header, f"Input (Z+1)", (col_width*1)+10, 20, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
        cv2.putText(header, f"Prediction (Z)", (col_width*2)+10, 20, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
        cv2.putText(header, f"Ground Truth (Z)", (col_width*3)+10, 20, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

        all_rows.append(np.vstack([header, row]))

    # Combine all rows and add a main title
    final_panel = np.vstack(all_rows)
    main_header = np.full((40, final_panel.shape[1], 3), 60, dtype=np.uint8)
    title = f"Slice Reconstruction - Batch #{batch_idx}, Middle Slice #{slice_idx}"
    cv2.putText(main_header, title, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
    
    return np.vstack([main_header, final_panel])


class BraTS2D5Dataset(Dataset):
    def __init__(self, data_dir, image_size, spacing, num_patients=None):
        self.image_size = image_size
        patient_dirs = sorted(glob.glob(os.path.join(data_dir, "BraTS*")))
        if num_patients is not None:
            print(f"--- Using a subset of {num_patients} patients for testing. ---")
            patient_dirs = patient_dirs[:num_patients]
        if not patient_dirs:
            raise FileNotFoundError(f"No patient data found in '{data_dir}'. Check your --data_dir path.")
        self.files = [{"t1": glob.glob(os.path.join(p, "*-t1n.nii.gz"))[0],"t1ce": glob.glob(os.path.join(p, "*-t1c.nii.gz"))[0],"t2": glob.glob(os.path.join(p, "*-t2w.nii.gz"))[0],"flair": glob.glob(os.path.join(p, "*-t2f.nii.gz"))[0],"label": glob.glob(os.path.join(p, "*-seg.nii.gz"))[0]} for p in patient_dirs]
        transforms = get_train_transforms(image_size, spacing)
        print("--- Pre-loading and processing volumes... ---")
        start_time = time()
        self.processed_volumes = []
        for i, patient_files in enumerate(self.files):
            self.processed_volumes.append(transforms(patient_files))
            if (i + 1) % 10 == 0 or (i + 1) == len(self.files):
                print(f"   Processed {i + 1}/{len(self.files)} patients...")
        print(f"--- Volume processing took {time() - start_time:.2f} seconds. ---")
        self.slice_map = []
        print("Mapping and filtering slices to create dataset...")
        for vol_idx, p_data in enumerate(self.processed_volumes):
            num_slices = p_data["label"].shape[3]
            # Ensure we have a valid context (prev and next slice)
            for slice_idx in range(1, num_slices - 1): 
                brain_slice = p_data["t1ce"][0, :, :, slice_idx]
                if torch.mean(brain_slice) > 0.1:
                    self.slice_map.append((vol_idx, slice_idx))
        print(f"Dataset ready. Found {len(self.slice_map)} valid slices from {len(self.files)} volumes.")

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

        # Return only the tensors needed for training and the slice index
        return input_tensor, target_tensor, slice_idx


def get_args():
    parser = argparse.ArgumentParser(description="2.5D Swin UNETR for Slice Reconstruction.")
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
    run_name = f"swin_unetr_reconstruction_{int(time())}"
    wandb.init(project="brats-2.5d-reconstruction", config=args, name=run_name)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)