"""
Command-line interface for OpenSpindleNet using Typer
"""
import typer
import numpy as np
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich import print as rprint

from openspindlenet import (
    detect_spindles, 
    SpindleInference, 
    Evaluator, 
    visualize_spindles,
    get_model_path,
    get_example_data_path
)

# Create Typer app
app = typer.Typer(help="OpenSpindleNet: Detect sleep spindles in EEG/iEEG data")
console = Console()

@app.command("detect")
def detect(
    input_file: Path = typer.Argument(..., help="Path to input EEG data file"),
    model_type: str = typer.Option("eeg", help="Model type to use: 'eeg' or 'ieeg'"),
    output_file: Optional[Path] = typer.Option(None, help="Path to save detection results"),
    visualize: bool = typer.Option(False, help="Visualize the detection results"),
    threshold: float = typer.Option(0.5, help="Confidence threshold for detections"),
    nms_threshold: float = typer.Option(0.3, help="Non-maximum suppression IoU threshold"),
):
    """Detect sleep spindles in EEG/iEEG data."""
    try:
        # Load data
        console.print(f"Loading data from [bold]{input_file}[/bold]...")
        try:
            data = np.loadtxt(input_file)
        except Exception as e:
            console.print(f"[bold red]Error loading data:[/bold red] {str(e)}")
            raise typer.Exit(code=1)
        
        # Detect spindles
        console.print(f"Detecting spindles using [bold]{model_type}[/bold] model...")
        try:
            results = detect_spindles(data, model_type=model_type)
        except Exception as e:
            console.print(f"[bold red]Error during detection:[/bold red] {str(e)}")
            raise typer.Exit(code=1)
        
        # Output results
        intervals = results["detection_intervals"]
        console.print(f"[bold green]Found {len(intervals)} spindle(s)[/bold green]")
        
        if output_file:
            # Save results
            np.savetxt(output_file, intervals, fmt='%.2f', header='start end')
            console.print(f"Results saved to [bold]{output_file}[/bold]")
        
        # Print results to console
        if len(intervals) > 0:
            console.print("Spindle intervals (seconds):")
            for i, (start, end) in enumerate(intervals):
                console.print(f"  {i+1}. {start:.2f} - {end:.2f} (duration: {end-start:.2f}s)")
        
        # Visualize if requested
        if visualize:
            console.print("Generating visualization...")
            visualize_spindles(data, intervals)
        
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        raise typer.Exit(code=1)

@app.command("info")
def info():
    """Display information about OpenSpindleNet."""
    console.print("[bold]OpenSpindleNet: Sleep Spindle Detection Tool[/bold]")
    console.print("\n[bold]Available models:[/bold]")
    
    try:
        eeg_model_path = get_model_path("eeg")
        console.print(f"  - EEG model: [green]{eeg_model_path}[/green]")
    except:
        console.print("  - EEG model: [red]Not found[/red]")
    
    try:
        ieeg_model_path = get_model_path("ieeg")
        console.print(f"  - iEEG model: [green]{ieeg_model_path}[/green]")
    except:
        console.print("  - iEEG model: [red]Not found[/red]")
    
    console.print("\n[bold]Example data:[/bold]")
    try:
        eeg_data_path = get_example_data_path("eeg")
        console.print(f"  - EEG example: [green]{eeg_data_path}[/green]")
    except:
        console.print("  - EEG example: [red]Not found[/red]")
    
    try:
        ieeg_data_path = get_example_data_path("ieeg")
        console.print(f"  - iEEG example: [green]{ieeg_data_path}[/green]")
    except:
        console.print("  - iEEG example: [red]Not found[/red]")

@app.command("example")
def run_example(
    data_type: str = typer.Option("eeg", help="Type of example data to use: 'eeg' or 'ieeg'"),
    visualize: bool = typer.Option(True, help="Visualize the detection results"),
):
    """Run spindle detection on example data."""
    try:
        # Get example data path
        console.print(f"Loading example {data_type} data...")
        try:
            data_path = get_example_data_path(data_type)
            data = np.loadtxt(data_path)
        except Exception as e:
            console.print(f"[bold red]Error loading example data:[/bold red] {str(e)}")
            raise typer.Exit(code=1)
        
        # Detect spindles
        console.print(f"Detecting spindles using {data_type} model...")
        try:
            results = detect_spindles(data, model_type=data_type)
        except Exception as e:
            console.print(f"[bold red]Error during detection:[/bold red] {str(e)}")
            raise typer.Exit(code=1)
        
        # Output results
        intervals = results["detection_intervals"]
        console.print(f"[bold green]Found {len(intervals)} spindle(s)[/bold green]")
        
        # Print results to console
        if len(intervals) > 0:
            console.print("Spindle intervals (seconds):")
            for i, (start, end) in enumerate(intervals):
                console.print(f"  {i+1}. {start:.2f} - {end:.2f} (duration: {end-start:.2f}s)")
        
        # Visualize if requested
        if visualize:
            console.print("Generating visualization...")
            visualize_spindles(data, intervals)
        
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        raise typer.Exit(code=1)

def main():
    """Entry point for the CLI."""
    app()

if __name__ == "__main__":
    app()
