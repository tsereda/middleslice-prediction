# train.py

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
from monai.losses import DiceCELoss
from transforms import get_train_transforms
from PIL import Image, ImageDraw, ImageFont

# --- NEW HELPER 1: Creates a 1x3 strip for a single modality ---
def create_modality_strip(
    inputs_sample, slice_idx, batch_idx, modality_name, modality_indices
):
    """Creates a 1x3 composite image: [Prev | Middle | Next]"""
    
    # Extract slices and convert to 8-bit images
    prev_slice = (inputs_sample[modality_indices["prev"]].numpy() * 255).astype(np.uint8)
    middle_slice = (inputs_sample[modality_indices["middle"]].numpy() * 255).astype(np.uint8)
    next_slice = (inputs_sample[modality_indices["next"]].numpy() * 255).astype(np.uint8)
    
    img_size = prev_slice.shape[1]
    canvas = Image.new("RGB", (img_size * 3, img_size + 40), (20, 20, 20)) # Dark canvas
    
    # Paste the three images side-by-side
    canvas.paste(Image.fromarray(prev_slice), (0, 40))
    canvas.paste(Image.fromarray(middle_slice), (img_size, 40))
    canvas.paste(Image.fromarray(next_slice), (img_size * 2, 40))

    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except IOError:
        font = ImageFont.load_default()

    # Main Title
    title = f"{modality_name.upper()} Slices - Batch #{batch_idx}, Middle Slice #{slice_idx}"
    draw.text((10, 10), title, fill="white", font=font)

    # Sub-labels
    draw.text((5, 45), f"Prev ({slice_idx-1})", fill="white", font=font)
    draw.text((img_size + 5, 45), f"Middle ({slice_idx})", fill="white", font=font)
    draw.text((img_size * 2 + 5, 45), f"Next ({slice_idx+1})", fill="white", font=font)
    
    return canvas

# --- NEW HELPER 2: Creates a 1x3 strip for segmentation results ---
def create_segmentation_strip(
    middle_slice_t1ce, gt_mask, pred_mask, slice_idx, batch_idx
):
    """Creates a 1x3 composite: [Anatomy (T1ce) | Ground Truth | Prediction]"""
    
    # Convert anatomy slice to 8-bit image
    anatomy_img = (middle_slice_t1ce.numpy() * 255).astype(np.uint8)
    
    # Scale masks to be visible (0-3 -> 0-255)
    gt_mask_img = (gt_mask * (255 / 3)).astype(np.uint8)
    pred_mask_img = (pred_mask * (255 / 3)).astype(np.uint8)

    img_size = anatomy_img.shape[1]
    canvas = Image.new("RGB", (img_size * 3, img_size + 40), (20, 20, 20)) # Dark canvas
    
    # Paste the three images side-by-side
    canvas.paste(Image.fromarray(anatomy_img), (0, 40))
    canvas.paste(Image.fromarray(gt_mask_img), (img_size, 40))
    canvas.paste(Image.fromarray(pred_mask_img), (img_size * 2, 40))
    
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except IOError:
        font = ImageFont.load_default()

    # Main Title
    title = f"Segmentation Output - Batch #{batch_idx}, Slice #{slice_idx}"
    draw.text((10, 10), title, fill="white", font=font)

    # Sub-labels
    draw.text((5, 45), "Anatomy (T1ce)", fill="white", font=font)
    draw.text((img_size + 5, 45), "Ground Truth", fill="white", font=font)
    draw.text((img_size * 2 + 5, 45), "Prediction", fill="white", font=font)
    
    return canvas


class BraTS2D5Dataset(Dataset):
    # ... (Dataset class is unchanged from the last optimized version) ...
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
        volume_idx, slice_idx = self.slice_map[index]
        patient_data = self.processed_volumes[volume_idx]
        img_modalities = torch.cat([patient_data['t1'], patient_data['t1ce'], patient_data['t2'], patient_data['flair']], dim=0)
        label_volume = patient_data['label']
        num_slices_in_vol = img_modalities.shape[3]
        prev_slice_idx = max(0, slice_idx - 1)
        next_slice_idx = min(num_slices_in_vol - 1, slice_idx + 1)
        stacked_slices = torch.stack([img_modalities[:, :, :, prev_slice_idx], img_modalities[:, :, :, slice_idx], img_modalities[:, :, :, next_slice_idx]], dim=0)
        in_channels = 4 * 3
        input_tensor = stacked_slices.view(in_channels, self.image_size[0], self.image_size[1])
        target_tensor = label_volume[:, :, :, slice_idx]
        return input_tensor, target_tensor, slice_idx


def get_args():
    parser = argparse.ArgumentParser(description="2.5D Swin UNETR training for BraTS.")
    parser.add_argument('--data_dir', type=str, required=True, help='Root directory for the BraTS dataset.')
    parser.add_argument('--output_dir', type=str, default='./checkpoints', help='Directory to save model checkpoints.')
    parser.add_argument('--epochs', type=int, default=25, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=4, help='Training batch size.')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate.')
    # --- MODIFIED: Default resolution changed to 256 ---
    parser.add_argument('--img_size', type=int, default=256, help='Image size (height and width).')
    parser.add_argument('--num_patients',type=int,default=None,help='Number of patient volumes to use for quick testing (default: all).')
    return parser.parse_args()


def main(args):
    torch.multiprocessing.set_sharing_strategy('file_system')
    run_name = f"swin_unetr_2.5d_{int(time())}"
    wandb.init(config=args, name=run_name)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    dataset = BraTS2D5Dataset(data_dir=args.data_dir, image_size=(args.img_size, args.img_size), spacing=(1.0, 1.0, 1.0), num_patients=args.num_patients)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=6)

    model = SwinUNETR(in_channels=12, out_channels=4, feature_size=24, spatial_dims=2).to(device)
    wandb.watch(model, log="all", log_freq=100)
    loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    print("Starting training...")
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        num_batches = len(data_loader)
        
        for i, (inputs, labels, slice_indices) in enumerate(data_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
            # --- MODIFIED: Logging 5 separate composite strips every 25 batches ---
            if (i + 1) % 25 == 0:
                print(f"  Epoch {epoch + 1}/{args.epochs}, Batch {i + 1}/{num_batches}...")
                
                inputs_sample = inputs[0].cpu()
                gt_mask = labels[0, 0].cpu().numpy()
                pred_mask = torch.argmax(outputs[0], dim=0).cpu().numpy()
                slice_idx_sample = slice_indices[0].item()

                modalities = { "t1": {"prev": 0, "middle": 4, "next": 8}, "t1ce": {"prev": 1, "middle": 5, "next": 9}, "t2": {"prev": 2, "middle": 6, "next": 10}, "flair": {"prev": 3, "middle": 7, "next": 11} }
                log_payload = {"batch_loss": loss.item()}

                # Log the 4 modality strips
                for name, indices in modalities.items():
                    modality_img = create_modality_strip(inputs_sample, slice_idx_sample, i + 1, name, indices)
                    log_payload[f"samples/{name}_strip"] = wandb.Image(modality_img)
                
                # Log the 1 segmentation strip
                t1ce_middle_slice = inputs_sample[modalities['t1ce']['middle']]
                seg_img = create_segmentation_strip(t1ce_middle_slice, gt_mask, pred_mask, slice_idx_sample, i + 1)
                log_payload["samples/segmentation_strip"] = wandb.Image(seg_img)

                wandb.log(log_payload)
        
        avg_loss = epoch_loss / num_batches
        print(f"--- Epoch {epoch + 1}/{args.epochs}, Average Loss: {avg_loss:.4f} ---")
        wandb.log({"epoch": epoch + 1, "avg_loss": avg_loss})
        
        checkpoint_path = os.path.join(args.output_dir, f"swin_unetr_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

    print("Training finished!")
    wandb.finish()


if __name__ == '__main__':
    args = get_args()
    main(args)