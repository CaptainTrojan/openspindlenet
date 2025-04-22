import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from matplotlib.backends.backend_pdf import PdfPages
from typing import List, Tuple, Optional, Union

def visualize_spindles(signal: np.ndarray, 
                       intervals: np.ndarray, 
                       segmentation: np.ndarray,
                       output_path: Optional[str] = None) -> plt.Figure:
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
    # For empty intervals, create a simple plot of the signal
    if len(intervals) == 0:
        fig = plt.figure(figsize=(12, 6))
        plt.plot(signal, color='gray', linewidth=0.8)
        plt.title("No spindles detected in signal", fontsize=14)
        plt.axis('off')
        
        # Save to PDF if output path is provided
        if output_path:
            # Ensure directory exists
            output_dir = os.path.dirname(os.path.abspath(output_path))
            if output_dir:  # Only create if there's an actual directory part
                os.makedirs(output_dir, exist_ok=True)
            
            # Save to PDF
            with PdfPages(output_path) as pdf:
                pdf.savefig(fig, bbox_inches='tight')
            
            print(f"Visualization saved to {os.path.abspath(output_path)}")
        
        return fig
        
    # Sort intervals by start position (chronological order)
    sorted_intervals = sorted(intervals, key=lambda x: x[0])
        
    # Create segmentation mask
    seq_len = len(signal)
    num_plots = len(sorted_intervals) + 2  # Full signal + segmentation + one per spindle
    
    # Figure setup - use a tall, wide figure
    fig = plt.figure(figsize=(12, 2 * num_plots))
    gs = gridspec.GridSpec(num_plots, 1, height_ratios=[1, 0.5] + [0.8] * len(sorted_intervals))
    
    # Plot 1: Full signal with highlighted spindles
    ax_full = plt.subplot(gs[0])
    # Plot the full signal with low alpha
    ax_full.plot(signal, color='gray', alpha=0.5, linewidth=0.8)
    
    # Highlight each spindle with higher alpha and thickness
    for i, interval in enumerate(sorted_intervals):
        # Handle cases where interval might be just [start, end] without confidence
        if len(interval) >= 3:
            start, end, conf = interval[0], interval[1], interval[2]
        else:
            start, end = interval[0], interval[1]
            conf = 0.5  # Default confidence if not provided
            
        start_idx, end_idx = int(start), int(end)
        ax_full.plot(range(start_idx, end_idx + 1), 
                     signal[start_idx:end_idx + 1], 
                     linewidth=2, 
                     alpha=0.9,
                     label=f"Spindle {i+1}")
        
        # Add confidence as text near each spindle
        mid_point = (start_idx + end_idx) // 2
        y_pos = signal[mid_point] + (0.5 * np.std(signal))
        ax_full.text(mid_point, y_pos, f"{conf:.2f}", fontsize=10, ha='center')
        
    # Remove axes from full signal plot
    ax_full.axis('off')
    
    # Plot 2: Segmentation mask
    ax_seg = plt.subplot(gs[1])
    ax_seg.plot(segmentation, color='blue', linewidth=1.5)
    # Add spindle numbers
    for i, interval in enumerate(sorted_intervals):
        start, end = interval[0], interval[1]
        mid_point = int((start + end) // 2)
        ax_seg.text(mid_point, 1.1, f"{i+1}", fontsize=12, ha='center', va='center')
    
    # Remove axes from segmentation plot
    ax_seg.axis('off')
    
    # Plots 3+: Zoomed-in spindles
    for i, interval in enumerate(sorted_intervals):
        # Handle cases where interval might be just [start, end] without confidence
        if len(interval) >= 3:
            start, end, conf = interval[0], interval[1], interval[2]
        else:
            start, end = interval[0], interval[1]
            conf = 0.5  # Default confidence if not provided
            
        start_idx, end_idx = int(start), int(end)
        
        # Add padding for better visibility (10% on each side)
        padding = int((end_idx - start_idx) * 0.1)
        padded_start = max(0, start_idx - padding)
        padded_end = min(len(signal) - 1, end_idx + padding)
        
        # Create subplot for this spindle
        ax_spindle = plt.subplot(gs[i+2])  # Offset by 2 now
        ax_spindle.plot(range(padded_start, padded_end + 1), 
                       signal[padded_start:padded_end + 1], 
                       linewidth=1.5)
        
        # Highlight the actual spindle section
        ax_spindle.axvspan(start_idx, end_idx, color='yellow', alpha=0.3)
        
        # Add spindle label
        ax_spindle.set_title(f"Spindle {i+1}: {conf:.2f}", fontsize=10)
        
        # Remove axes
        ax_spindle.axis('off')
    
    # Tight layout to minimize whitespace
    plt.tight_layout()
    
    # Save to PDF if output path is provided
    if output_path:
        # Ensure directory exists
        output_dir = os.path.dirname(os.path.abspath(output_path))
        if output_dir:  # Only create if there's an actual directory part
            os.makedirs(output_dir, exist_ok=True)
        
        # Save to PDF
        with PdfPages(output_path) as pdf:
            pdf.savefig(fig, bbox_inches='tight')
        
        print(f"Visualization saved to {os.path.abspath(output_path)}")
    
    return fig
