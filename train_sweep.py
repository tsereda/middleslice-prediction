# train_sweep.py - W&B Sweep version supporting multiple architectures

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
from monai.networks.nets import SwinUNETR, UNETR, BasicUNet
from torch.nn import L1Loss 
from transforms import get_train_transforms
from logging_utils import create_reconstruction_log_panel
from torchmetrics.image import StructuralSimilarityIndexMeasure
from skimage.metrics import structural_similarity as ssim_3d
from skimage.metrics import peak_signal_noise_ratio as psnr_3d


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
        
        self.files = []
        skipped_patients = 0
        
        for p in patient_dirs:
            t1_patterns = ["*-t1n.nii*", "*_t1n.nii*"]
            t1ce_patterns = ["*-t1c.nii*", "*_t1c.nii*"]
            t2_patterns = ["*-t2w.nii*", "*_t2w.nii*"]
            flair_patterns = ["*-t2f.nii*", "*_t2f.nii*"]
            seg_patterns = ["*-seg.nii*", "*_seg.nii*"]
            
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
            
            if not all([t1_files, t1ce_files, t2_files, flair_files, seg_files]):
                print(f"Warning: Missing files in {os.path.basename(p)}")
                print(f"  T1N: {len(t1_files)}, T1C: {len(t1ce_files)}, T2W: {len(t2_files)}, T2F: {len(flair_files)}, SEG: {len(seg_files)}")
                all_nii_files = glob.glob(os.path.join(p, "*.nii*"))
                print(f"  All .nii* files found: {[os.path.basename(f) for f in all_nii_files]}")
                skipped_patients += 1
                continue
                
            self.files.append({
                "t1": t1_files[0],
                "t1ce": t1ce_files[0],
                "t2": t2_files[0],
                "flair": flair_files[0],
                "label": seg_files[0]
            })
        
        if not self.files:
            raise ValueError(f"No valid patient data found. All {len(patient_dirs)} directories were missing required files.")
            
        if skipped_patients > 0:
            print(f"Note: Skipped {skipped_patients} patients due to missing files. Using {len(self.files)} patients.")
        
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
        
        self.slice_map = []
        print("Mapping all available slices to create dataset...")
        for vol_idx, p_data in enumerate(self.processed_volumes):
            num_slices = p_data["label"].shape[3]
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


def compute_psnr(pred, target, data_range=1.0):
    """
    Compute PSNR between prediction and target.
    Args:
        pred: Predicted tensor
        target: Ground truth tensor
        data_range: Maximum possible pixel value
    """
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(data_range / torch.sqrt(mse))


def evaluate_3d_volume(model, dataset, volume_idx, device):
    """
    Reconstruct a full 3D volume and compute 3D metrics.
    
    Args:
        model: Trained model
        dataset: BraTS2D5Dataset instance
        volume_idx: Index of volume to reconstruct
        device: Device for computation
    
    Returns:
        dict with metrics: mae_3d, psnr_3d, ssim_3d
    """
    model.eval()
    patient_data = dataset.processed_volumes[volume_idx]
    num_slices = patient_data["label"].shape[3]
    
    # Get volume dimensions
    c, h, w, d = patient_data['t1'].shape
    
    # Initialize arrays for predicted and ground truth volumes (4 modalities)
    pred_volume = np.zeros((4, h, w, num_slices), dtype=np.float32)
    gt_volume = np.zeros((4, h, w, num_slices), dtype=np.float32)
    
    # Concatenate all modalities
    img_modalities = torch.cat([
        patient_data['t1'], 
        patient_data['t1ce'], 
        patient_data['t2'], 
        patient_data['flair']
    ], dim=0)
    
    with torch.no_grad():
        for slice_idx in range(1, num_slices - 1):
            # Prepare input
            prev_slice = img_modalities[:, :, :, slice_idx - 1]
            next_slice = img_modalities[:, :, :, slice_idx + 1]
            input_tensor = torch.cat([prev_slice, next_slice], dim=0).unsqueeze(0).to(device)
            
            # Get prediction
            output = model(input_tensor)
            
            # Store prediction and ground truth
            pred_volume[:, :, :, slice_idx] = output.squeeze(0).cpu().numpy()
            gt_volume[:, :, :, slice_idx] = img_modalities[:, :, :, slice_idx].cpu().numpy()
    
    # Handle boundary slices (copy from neighbors or set to ground truth)
    pred_volume[:, :, :, 0] = gt_volume[:, :, :, 0]
    pred_volume[:, :, :, -1] = gt_volume[:, :, :, -1]
    
    # Compute 3D metrics
    mae_3d = np.mean(np.abs(pred_volume - gt_volume))
    
    # Compute PSNR and SSIM per modality, then average
    psnr_per_modality = []
    ssim_per_modality = []
    
    for mod_idx in range(4):
        pred_mod = pred_volume[mod_idx]
        gt_mod = gt_volume[mod_idx]
        
        # PSNR
        data_range = gt_mod.max() - gt_mod.min()
        if data_range > 0:
            psnr_val = psnr_3d(gt_mod, pred_mod, data_range=data_range)
            psnr_per_modality.append(psnr_val)
        
        # SSIM (3D)
        ssim_val = ssim_3d(gt_mod, pred_mod, data_range=data_range)
        ssim_per_modality.append(ssim_val)
    
    return {
        'mae_3d': mae_3d,
        'psnr_3d': np.mean(psnr_per_modality) if psnr_per_modality else 0.0,
        'ssim_3d': np.mean(ssim_per_modality)
    }


def create_model(model_type, feature_size=24, device='cuda'):
    """
    Create model based on type string.
    
    Args:
        model_type: One of 'swin_unetr', 'unetr', or 'basic_unet'
        feature_size: Feature size for the model
        device: Device to move model to
    
    Returns:
        model: Instantiated model on specified device
    """
    in_channels = 8  # 4 modalities × 2 slices (prev + next)
    out_channels = 4  # 4 modalities output
    
    if model_type == 'swin_unetr':
        print(f"Creating SwinUNETR with feature_size={feature_size}")
        model = SwinUNETR(
            in_channels=in_channels,
            out_channels=out_channels,
            feature_size=feature_size,
            spatial_dims=2
        )
    elif model_type == 'unetr':
        print(f"Creating UNETR with feature_size={feature_size}")
        model = UNETR(
            in_channels=in_channels,
            out_channels=out_channels,
            img_size=(256, 256),  # Default, will match actual input
            feature_size=feature_size,
            spatial_dims=2
        )
    elif model_type == 'basic_unet':
        print(f"Creating BasicUNet with features={feature_size}")
        model = BasicUNet(
            spatial_dims=2,
            in_channels=in_channels,
            out_channels=out_channels,
            features=(feature_size, feature_size*2, feature_size*4, feature_size*8, feature_size*16, feature_size*16)
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}. Must be one of: swin_unetr, unetr, basic_unet")
    
    return model.to(device)


def get_args():
    parser = argparse.ArgumentParser(description="2.5D Model Sweep for Slice Reconstruction - BraTS 2023")
    parser.add_argument('--data_dir', type=str, default='BraTS_126_samples', help='Root directory for the BraTS dataset.')
    parser.add_argument('--output_dir', type=str, default='./checkpoints', help='Directory to save model checkpoints.')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=8, help='Training batch size.')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate.')
    parser.add_argument('--img_size', type=int, default=256, help='Image size (height and width).')
    parser.add_argument('--feature_size', type=int, default=24, help='Feature size for model.')
    parser.add_argument('--num_patients', type=int, default=None, help='Number of patient volumes to use for quick testing (default: all).')
    parser.add_argument('--model_type', type=str, default='swin_unetr', choices=['swin_unetr', 'unetr', 'basic_unet'], 
                        help='Model architecture to use (overridden by W&B sweep).')
    return parser.parse_args()


def main(args):
    torch.multiprocessing.set_sharing_strategy('file_system')
    
    # Initialize W&B - will automatically pull sweep config if running in sweep
    run_name = f"{wandb.config.model_type if wandb.run else args.model_type}_brats2023_{int(time())}"
    wandb.init(project="brats2023-architecture-comparison", config=vars(args), name=run_name)
    
    # Use W&B config if available (in sweep), otherwise use CLI args
    config = wandb.config
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Model type: {config.model_type}")
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize 2D metrics
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    # Create dataset
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

    # Create model based on sweep config or CLI arg
    model = create_model(config.model_type, feature_size=args.feature_size, device=device)
    wandb.watch(model, log="all", log_freq=100)
    
    loss_function = L1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    print(f"Starting training for {config.model_type}...")
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        epoch_ssim = 0
        epoch_psnr = 0
        num_batches = len(data_loader)
        
        for i, (inputs, targets, slice_indices) in enumerate(data_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, targets)
            loss.backward()
            optimizer.step()
            
            # Compute 2D metrics
            with torch.no_grad():
                batch_ssim = ssim_metric(outputs, targets)
                batch_psnr = compute_psnr(outputs, targets, data_range=1.0)
            
            epoch_loss += loss.item()
            epoch_ssim += batch_ssim.item()
            epoch_psnr += batch_psnr.item()
            
            if (i + 1) % 100 == 0:
                print(f"   Epoch {epoch + 1}/{args.epochs}, Batch {i + 1}/{num_batches} | "
                      f"L1: {loss.item():.4f}, SSIM: {batch_ssim.item():.4f}, PSNR: {batch_psnr.item():.2f} dB")
                
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
                    "batch_ssim": batch_ssim.item(),
                    "batch_psnr": batch_psnr.item(),
                    "reconstruction_samples": wandb.Image(panel_rgb)
                })
        
        # Compute epoch averages
        avg_loss = epoch_loss / num_batches
        avg_ssim = epoch_ssim / num_batches
        avg_psnr = epoch_psnr / num_batches
        
        print(f"--- Epoch {epoch + 1}/{args.epochs} ---")
        print(f"Average L1 Loss: {avg_loss:.4f}, SSIM: {avg_ssim:.4f}, PSNR: {avg_psnr:.2f} dB")
        
        # 3D volume evaluation (every 5 epochs or last epoch)
        if (epoch + 1) % 5 == 0 or (epoch + 1) == args.epochs:
            print("Running 3D volume evaluation...")
            num_eval_volumes = min(3, len(dataset.processed_volumes))  # Evaluate 3 volumes
            
            vol_metrics_list = []
            for vol_idx in range(num_eval_volumes):
                vol_metrics = evaluate_3d_volume(model, dataset, vol_idx, device)
                vol_metrics_list.append(vol_metrics)
                print(f"  Volume {vol_idx}: MAE={vol_metrics['mae_3d']:.4f}, "
                      f"PSNR={vol_metrics['psnr_3d']:.2f} dB, SSIM={vol_metrics['ssim_3d']:.4f}")
            
            # Average 3D metrics
            avg_3d_metrics = {
                'mae_3d': np.mean([m['mae_3d'] for m in vol_metrics_list]),
                'psnr_3d': np.mean([m['psnr_3d'] for m in vol_metrics_list]),
                'ssim_3d': np.mean([m['ssim_3d'] for m in vol_metrics_list])
            }
            
            wandb.log({
                "epoch": epoch + 1,
                "avg_epoch_l1_loss": avg_loss,
                "avg_epoch_ssim": avg_ssim,
                "avg_epoch_psnr": avg_psnr,
                "avg_3d_mae": avg_3d_metrics['mae_3d'],
                "avg_3d_psnr": avg_3d_metrics['psnr_3d'],
                "avg_3d_ssim": avg_3d_metrics['ssim_3d']
            })
        else:
            wandb.log({
                "epoch": epoch + 1,
                "avg_epoch_l1_loss": avg_loss,
                "avg_epoch_ssim": avg_ssim,
                "avg_epoch_psnr": avg_psnr
            })
        
        # Save checkpoint with model type in filename
        checkpoint_path = os.path.join(args.output_dir, f"{config.model_type}_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

    print("Training finished!")
    wandb.finish()


if __name__ == '__main__':
    args = get_args()
    main(args)
