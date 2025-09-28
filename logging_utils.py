import numpy as np
import cv2
import torch
from typing import List

def create_reconstruction_log_panel(
    inputs_sample: torch.Tensor,      # Model input (Prev/Next slices), shape [8, H, W]
    target_sample: torch.Tensor,      # Ground Truth (Real middle slice), shape [4, H, W]
    output_sample: torch.Tensor,      # Model Prediction (Reconstructed middle slice), shape [4, H, W]
    slice_idx: int,
    batch_idx: int
) -> np.ndarray:
    modalities = ["t1", "t1ce", "t2", "flair"]
    all_rows = []
    header_height = 30
    
    for i, name in enumerate(modalities):
        prev_slice = (inputs_sample[i].cpu().numpy() * 255).astype(np.uint8)
        next_slice = (inputs_sample[i + 4].cpu().numpy() * 255).astype(np.uint8)
        
        gt_middle_float = target_sample[i].cpu().numpy()
        pred_middle_float = output_sample[i].cpu().numpy()
        
        pred_middle_clipped_scaled = (np.clip(pred_middle_float, 0, 1) * 255).astype(np.uint8)
        gt_middle_scaled = (gt_middle_float * 255).astype(np.uint8)

        abs_diff_float = np.abs(pred_middle_float - gt_middle_float)
        
        max_diff_for_viz = 0.4 
        scaled_diff = np.clip((abs_diff_float / max_diff_for_viz) * 255, 0, 255)
        abs_diff_uint8 = scaled_diff.astype(np.uint8)
        
        abs_diff_bgr = cv2.applyColorMap(abs_diff_uint8, cv2.COLORMAP_HOT)
        
        prev_bgr = cv2.cvtColor(prev_slice, cv2.COLOR_GRAY2BGR)
        next_bgr = cv2.cvtColor(next_slice, cv2.COLOR_GRAY2BGR)
        pred_bgr = cv2.cvtColor(pred_middle_clipped_scaled, cv2.COLOR_GRAY2BGR)
        gt_bgr = cv2.cvtColor(gt_middle_scaled, cv2.COLOR_GRAY2BGR)

        row = np.hstack([prev_bgr, next_bgr, pred_bgr, gt_bgr, abs_diff_bgr])
        
        header = np.full((header_height, row.shape[1], 3), 40, dtype=np.uint8)
        
        cv2.putText(header, f"{name.upper()}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        col_width = prev_bgr.shape[1]
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_color = (200, 200, 200)
        thickness = 1
        column_titles = ["Input (Z-1)", "Input (Z+1)", "Prediction (Z)", "Ground Truth (Z)", "Abs Difference"]

        for j, title in enumerate(column_titles):
            (text_width, _), _ = cv2.getTextSize(title, font, font_scale, thickness)
            text_x = (col_width * j) + ((col_width - text_width) // 2)
            cv2.putText(header, title, (text_x, 20), font, font_scale, font_color, thickness)

        all_rows.append(np.vstack([header, row]))

    final_panel = np.vstack(all_rows)
    main_header = np.full((40, final_panel.shape[1], 3), 60, dtype=np.uint8)
    title = f"Slice Reconstruction - Batch #{batch_idx}, Middle Slice #{slice_idx}"
    cv2.putText(main_header, title, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
    
    return np.vstack([main_header, final_panel])