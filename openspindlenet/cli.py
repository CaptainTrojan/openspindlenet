import argparse
import os
import sys
import numpy as np  # Add the missing import
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint
from typing import List, Optional

from .inference import detect_spindles
from .visualization import visualize_spindles
from . import get_model_path, get_example_data_path
import traceback

console = Console()

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="OpenSpindleNet - Neural sleep spindle detection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        "data_file", 
        help="Path to the data file containing EEG/iEEG signal"
    )
    
    # Optional arguments
    parser.add_argument(
        "--model", 
        choices=["eeg", "ieeg"], 
        default="eeg", 
        help="Type of model to use"
    )
    parser.add_argument(
        "--custom-model", 
        help="Path to custom ONNX model file"
    )
    parser.add_argument(
        "--vis", 
        action="store_true", 
        help="Generate visualization PDF"
    )
    parser.add_argument(
        "--output", 
        help="Path to save visualization (defaults to <data_file>_spindles.pdf)"
    )
    parser.add_argument(
        "--threshold", 
        type=float, 
        default=0.5, 
        help="Confidence threshold for spindle detection"
    )
    
    # Example data
    parser.add_argument(
        "--example", 
        action="store_true",
        help="Use example data instead of providing a file"
    )
    
    return parser.parse_args()

def display_results(results, threshold=0.5):
    """Display detection results in a rich table."""
    if 'detection_intervals' not in results or len(results['detection_intervals']) == 0:
        console.print("[yellow]No spindles detected.[/yellow]")
        return
        
    intervals = results['detection_intervals']
    
    # Create table for results
    table = Table(title=f"Detected Spindles (threshold={threshold})")
    
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Start (sample)", style="magenta")
    table.add_column("End (sample)", style="magenta")
    table.add_column("Duration (samples)", style="green")
    table.add_column("Confidence", style="yellow")
    
    for i, (start, end, conf) in enumerate(intervals):
        if conf >= threshold:
            table.add_row(
                f"{i+1}", 
                f"{start:.1f}", 
                f"{end:.1f}", 
                f"{(end-start):.1f}", 
                f"{conf:.4f}"
            )
    
    console.print(table)
    
    # Summary
    console.print(Panel(
        f"[bold green]Total spindles detected: {len(intervals)}[/bold green]\n"
        f"Average duration: {np.mean(intervals[:, 1] - intervals[:, 0]):.1f} samples\n"
        f"Average confidence: {np.mean(intervals[:, 2]):.4f}"
    ))

def main():
    """Main entry point for CLI."""
    args = parse_arguments()
    
    try:
        console.print("[bold blue]OpenSpindleNet[/bold blue] - Neural sleep spindle detection", justify="center")
        
        # Determine file path - use example or provided
        if args.example:
            data_path = get_example_data_path(args.model)
            console.print(f"Using example {args.model} data: {data_path}")
        else:
            data_path = args.data_file
            if not os.path.exists(data_path):
                console.print(f"[bold red]Error:[/bold red] File not found: {data_path}")
                return 1
        
        with console.status("[bold green]Processing...[/bold green]"):
            # Run spindle detection
            results = detect_spindles(
                data_path, 
                model_type=args.model, 
                custom_model_path=args.custom_model
            )
        
        # Display results
        display_results(results, args.threshold)
        
        # Generate visualization if requested
        if args.vis:
            if len(results['detection_intervals']) > 0:
                output_path = args.output
                if not output_path:
                    base_name = os.path.splitext(os.path.basename(data_path))[0]
                    output_path = f"{base_name}_spindles.pdf"
                
                console.print(f"[green]Generating visualization...[/green]")
                visualize_spindles(
                    results['raw_signal'],
                    results['detection_intervals'],
                    results['segmentation'],
                    output_path
                )
                console.print(f"[green]Visualization saved to: {output_path}[/green]")
            else:
                console.print("[yellow]No spindles detected, skipping visualization.[/yellow]")
        
        return 0
        
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        console.print("[bold red]Stack trace:[/bold red]")
        console.print(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main())
