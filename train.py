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

# --- BraTS2D5Dataset class from the previous step remains unchanged ---
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
        return input_tensor, target_tensor

# --- get_args function remains unchanged ---
def get_args():
    parser = argparse.ArgumentParser(description="2.5D Swin UNETR training for BraTS.")
    parser.add_argument('--data_dir', type=str, required=True, help='Root directory for the BraTS dataset.')
    parser.add_argument('--output_dir', type=str, default='./checkpoints', help='Directory to save model checkpoints.')
    parser.add_argument('--epochs', type=int, default=25, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=4, help='Training batch size.')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate.')
    parser.add_argument('--img_size', type=int, default=128, help='Image size (height and width).')
    parser.add_argument('--num_patients',type=int,default=None,help='Number of patient volumes to use for quick testing (default: all).')
    return parser.parse_args()


def main(args):
    torch.multiprocessing.set_sharing_strategy('file_system')

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
            
            # --- MODIFIED: Log metrics and separate image samples every 25 batches ---
            if (i + 1) % 25 == 0:
                print(f"  Epoch {epoch + 1}/{args.epochs}, Batch {i + 1}/{num_batches}...")
                
                # Detach tensors from the graph and move to CPU for logging
                inputs_sample = inputs[0].cpu()
                labels_sample = labels[0].cpu()
                outputs_sample = outputs[0].cpu()

                # Get the prediction mask by taking argmax of model output
                pred_mask = torch.argmax(outputs_sample, dim=0).numpy()
                gt_mask = labels_sample[0].numpy()

                # Define the channel mapping for clarity
                modalities = ['t1', 't1ce', 't2', 'flair']
                
                # Channel indices for [previous, current, next] slices
                # Prev: 0-3, Curr: 4-7, Next: 8-11
                log_payload = {"batch_loss": loss.item()}

                # Log Previous Slice (4 modalities)
                for m_idx, modality in enumerate(modalities):
                    log_payload[f"samples/previous_slice/{modality}"] = wandb.Image(inputs_sample[m_idx])
                
                # Log Next Slice (4 modalities)
                for m_idx, modality in enumerate(modalities):
                    log_payload[f"samples/next_slice/{modality}"] = wandb.Image(inputs_sample[m_idx + 8])

                # Log Middle Slice Ground Truth and Prediction
                log_payload["samples/middle_slice/ground_truth"] = wandb.Image(gt_mask, caption="Ground Truth Mask")
                log_payload["samples/middle_slice/prediction"] = wandb.Image(pred_mask, caption="Predicted Mask")
                
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