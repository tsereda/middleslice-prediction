import torch
import numpy as np
import os
import glob
import nibabel as nib

def quick_data_check(data_dir, num_patients=3):
    """
    Quick check of raw vs processed data without using the transforms module
    """
    print("=== QUICK DATA ANALYSIS ===")
    
    # Get patient directories
    patient_dirs = sorted(glob.glob(os.path.join(data_dir, "BraTS*")))[:num_patients]
    
    if not patient_dirs:
        print(f"No patient data found in '{data_dir}'")
        return
    
    print(f"Analyzing {len(patient_dirs)} patients...")
    
    for i, patient_dir in enumerate(patient_dirs):
        print(f"\nPatient {i+1}: {os.path.basename(patient_dir)}")
        
        # Find T1CE file (most commonly used)
        t1ce_files = glob.glob(os.path.join(patient_dir, "*_t1ce.nii*"))
        if not t1ce_files:
            print("  No T1CE file found")
            continue
            
        t1ce_file = t1ce_files[0]
        print(f"  T1CE file: {os.path.basename(t1ce_file)}")
        
        # Load raw data
        try:
            img = nib.load(t1ce_file)
            data = img.get_fdata()
            print(f"  Raw data shape: {data.shape}")
            print(f"  Raw data range: {data.min():.2f} to {data.max():.2f}")
            print(f"  Raw data mean: {data.mean():.2f}")
            
            # Check a middle slice
            if len(data.shape) >= 3:
                middle_slice_idx = data.shape[2] // 2
                middle_slice = data[:, :, middle_slice_idx]
                print(f"  Middle slice (idx {middle_slice_idx}) mean: {middle_slice.mean():.6f}")
                print(f"  Middle slice max: {middle_slice.max():.6f}")
                
                # Count non-zero pixels
                non_zero_pixels = np.sum(middle_slice > 0)
                total_pixels = middle_slice.size
                print(f"  Non-zero pixels: {non_zero_pixels}/{total_pixels} ({100*non_zero_pixels/total_pixels:.1f}%)")
                
        except Exception as e:
            print(f"  Error loading file: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='Root directory for the BraTS dataset.')
    parser.add_argument('--num_patients', type=int, default=3, help='Number of patients to analyze')
    args = parser.parse_args()
    
    quick_data_check(args.data_dir, args.num_patients)