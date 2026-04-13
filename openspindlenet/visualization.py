import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from matplotlib.backends.backend_pdf import PdfPages
from typing import Optional


def _segmentation_to_intervals(segmentation: np.ndarray) -> np.ndarray:
    seg = np.asarray(segmentation).astype(np.int32)
    if seg.ndim == 2:
        seg = seg[:, 0]
    seg = (seg > 0).astype(np.int32)

    starts = np.where(np.diff(seg) == 1)[0]
    ends = np.where(np.diff(seg) == -1)[0]

    if seg[0] == 1:
        starts = np.concatenate([[0], starts])
    if seg[-1] == 1:
        ends = np.concatenate([ends, [len(seg) - 1]])

    intervals = []
    for s, e in zip(starts, ends):
        intervals.append([float(s), float(e), 1.0])
    if not intervals:
        return np.zeros((0, 3), dtype=np.float32)
    return np.asarray(intervals, dtype=np.float32)

def visualize_spindles(signal: np.ndarray, 
                       intervals: np.ndarray, 
                       segmentation: np.ndarray,
                       output_path: Optional[str] = None,
                       true_segmentation: Optional[np.ndarray] = None,
                       metrics: Optional[dict] = None,
                       title: Optional[str] = None,
                       close_figure: bool = False) -> plt.Figure:
    """
    Visualize the signal with detected spindles.
    
    Args:
        signal: The raw EEG signal
        intervals: The detected spindle intervals (start, end, confidence)
        segmentation: Binary array marking spindle locations
        output_path: Path to save the PDF visualization (optional)
        
    Returns:
        The matplotlib figure object
    """
    intervals = np.asarray(intervals)
    # Sort intervals by start position (chronological order)
    sorted_intervals = sorted(intervals, key=lambda x: x[0]) if len(intervals) > 0 else []

    true_intervals = []
    if true_segmentation is not None:
        true_intervals = sorted(_segmentation_to_intervals(true_segmentation), key=lambda x: x[0])
        
    # Create segmentation mask
    num_plots = len(sorted_intervals) + (3 if true_segmentation is not None else 2)
    
    # Figure setup - use a tall, wide figure
    fig = plt.figure(figsize=(12, 2 * num_plots))
    first_rows = [1.3, 0.5, 0.5] if true_segmentation is not None else [1.3, 0.5]
    gs = gridspec.GridSpec(num_plots, 1, height_ratios=first_rows + [0.8] * len(sorted_intervals))
    
    # Plot 1: Full signal with highlighted predictions and optional labels
    ax_full = plt.subplot(gs[0])
    ax_full.plot(signal, color='gray', alpha=0.5, linewidth=0.8)
    
    # Highlight each spindle with higher alpha and thickness
    for i, interval in enumerate(sorted_intervals):
        if len(interval) >= 3:
            start, end, conf = interval[0], interval[1], interval[2]
        else:
            start, end = interval[0], interval[1]
            conf = 0.5
            
        start_idx, end_idx = int(start), int(end)
        ax_full.plot(range(start_idx, end_idx + 1), 
                     signal[start_idx:end_idx + 1], 
                     linewidth=2, 
                     alpha=0.9,
                     label=f"Spindle {i+1}")
        
        mid_point = (start_idx + end_idx) // 2
        y_pos = signal[mid_point] + (0.5 * np.std(signal))
        ax_full.text(mid_point, y_pos, f"{conf:.2f}", fontsize=10, ha='center')

    for interval in true_intervals:
        start_idx, end_idx = int(interval[0]), int(interval[1])
        ax_full.axvspan(start_idx, end_idx, color='green', alpha=0.1)

    if metrics:
        det = metrics.get("detection", {})
        seg = metrics.get("segmentation", {})
        metrics_text = (
            f"Det F1={det.get('f1', 0.0):.3f} "
            f"(P={det.get('precision', 0.0):.3f}, R={det.get('recall', 0.0):.3f})\n"
            f"Seg F1={seg.get('f1', 0.0):.3f} "
            f"(P={seg.get('precision', 0.0):.3f}, R={seg.get('recall', 0.0):.3f})"
        )
        ax_full.text(
            0.01,
            0.97,
            metrics_text,
            transform=ax_full.transAxes,
            fontsize=10,
            va='top',
            ha='left',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none')
        )

    if title:
        ax_full.set_title(title, fontsize=12)
    else:
        ax_full.set_title("Predictions (yellow) and labels (green)", fontsize=12)
        
    # Remove axes from full signal plot
    ax_full.axis('off')
    
    # Plot 2: Predicted segmentation mask
    ax_seg = plt.subplot(gs[1])
    ax_seg.plot(segmentation, color='blue', linewidth=1.5)
    ax_seg.set_title("Predicted segmentation", fontsize=10)
    for i, interval in enumerate(sorted_intervals):
        start, end = interval[0], interval[1]
        mid_point = int((start + end) // 2)
        ax_seg.text(mid_point, 1.1, f"{i+1}", fontsize=12, ha='center', va='center')
    ax_seg.axis('off')

    row_offset = 2
    if true_segmentation is not None:
        ax_true = plt.subplot(gs[2])
        ax_true.plot(true_segmentation, color='green', linewidth=1.5)
        ax_true.set_title("Label segmentation", fontsize=10)
        ax_true.axis('off')
        row_offset = 3
    
    # Following plots: zoomed-in predicted intervals
    for i, interval in enumerate(sorted_intervals):
        if len(interval) >= 3:
            start, end, conf = interval[0], interval[1], interval[2]
        else:
            start, end = interval[0], interval[1]
            conf = 0.5
            
        start_idx, end_idx = int(start), int(end)
        
        padding = int((end_idx - start_idx) * 0.1)
        padded_start = max(0, start_idx - padding)
        padded_end = min(len(signal) - 1, end_idx + padding)
        
        ax_spindle = plt.subplot(gs[i + row_offset])
        ax_spindle.plot(range(padded_start, padded_end + 1), 
                       signal[padded_start:padded_end + 1], 
                       linewidth=1.5)
        
        # Highlight the actual spindle section
        ax_spindle.axvspan(start_idx, end_idx, color='yellow', alpha=0.3)
        
        # Add spindle label
        ax_spindle.set_title(f"Spindle {i+1}: {conf:.2f}", fontsize=10)
        
        # Remove axes
        ax_spindle.axis('off')
    
    plt.tight_layout()
    
    if output_path:
        output_dir = os.path.dirname(os.path.abspath(output_path))
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        with PdfPages(output_path) as pdf:
            pdf.savefig(fig, bbox_inches='tight')

    if close_figure:
        plt.close(fig)
    
    return fig
