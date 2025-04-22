"""
Command-line interface for OpenSpindleNet using Typer
"""
import importlib.util
import sys
from pathlib import Path
import os
import datetime

# Check for optional CLI dependencies
CLI_AVAILABLE = True
MISSING_PACKAGES = []

for package in ["typer", "rich"]:
    if importlib.util.find_spec(package) is None:
        CLI_AVAILABLE = False
        MISSING_PACKAGES.append(package)

if not CLI_AVAILABLE:
    def main():
        """Entry point for the CLI when dependencies are missing."""
        print("Error: OpenSpindleNet CLI dependencies are not installed.")
        print(f"Missing packages: {', '.join(MISSING_PACKAGES)}")
        print("\nTo use the CLI features, install the package with CLI extras:")
        print("    pip install openspindlenet[cli]")
        sys.exit(1)
else:
    import typer
    import numpy as np
    from typing import Optional
    from rich.console import Console
    from rich import print as rprint

    # Import from internal modules - note we use the renamed detect function
    from . import detect
    from . import get_model_path, get_example_data_path
    from .visualization import visualize_spindles

    # Create Typer app with more specific help text
    app = typer.Typer(
        help="OpenSpindleNet: Detect sleep spindles in EEG/iEEG data\n\n"
             "Basic usage:\n"
             "  spindle-detect detect <FILE_PATH>  # Detect spindles in a file\n"
             "  spindle-detect example            # Run detection on example data\n"
             "  spindle-detect info               # Show information about available models"
    )
    console = Console()

    def generate_output_filename(prefix, suffix=None):
        """Generate a timestamped output filename."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        if suffix:
            return f"{prefix}_{timestamp}_{suffix}.pdf"
        return f"{prefix}_{timestamp}.pdf"

    @app.command("detect")
    def detect_cmd(
        input_file: Path = typer.Argument(..., help="Path to input EEG data file"),
        model_type: str = typer.Option("eeg", help="Model type to use: 'eeg' or 'ieeg'"),
        output_file: Optional[Path] = typer.Option(None, help="Path to save detection results"),
        visualization_file: Optional[Path] = typer.Option(None, help="Path to save visualization PDF (default: auto-generated)"),
        visualize: bool = typer.Option(False, "--visualize/--no-visualize", help="Visualize the detection results"),
        threshold: float = typer.Option(0.5, help="Confidence threshold for detections"),
        nms_threshold: float = typer.Option(0.3, help="Non-maximum suppression IoU threshold"),
    ):
        """Detect sleep spindles in EEG/iEEG data file."""
        try:
            # Check if user might have intended to use the example command
            if str(input_file) == "example":
                console.print("[bold yellow]Did you mean to use the example command?[/bold yellow]")
                console.print("Try: spindle-detect example")
                raise typer.Exit(code=1)
                
            # Load data
            console.print(f"Loading data from [bold]{input_file}[/bold]...")
            try:
                if not input_file.exists():
                    console.print(f"[bold red]Error:[/bold red] File '{input_file}' does not exist")
                    raise typer.Exit(code=1)
                data = np.loadtxt(input_file)
            except Exception as e:
                console.print(f"[bold red]Error loading data:[/bold red] {str(e)}")
                raise typer.Exit(code=1)
            
            # Detect spindles
            console.print(f"Detecting spindles using [bold]{model_type}[/bold] model...")
            try:
                results = detect(data, model_type=model_type)
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
                # Print each interval - safely handle both [start, end] and [start, end, confidence] formats
                for i, interval in enumerate(intervals):
                    if len(interval) >= 2:  # Make sure we have at least start and end
                        start, end = interval[0], interval[1]
                        console.print(f"  {i+1}. {start:.2f} - {end:.2f} (duration: {end-start:.2f}s)")
            
            # Visualize if requested
            if visualize:
                console.print("Generating visualization...")
                
                # Auto-generate visualization filename if not provided
                if visualization_file is None:
                    input_stem = Path(input_file).stem
                    visualization_file = Path(generate_output_filename(f"spindles_{input_stem}"))
                
                # Create a simple segmentation mask for visualization
                # (1 for detected spindle segments, 0 elsewhere)
                segmentation = np.zeros(len(data))
                
                # Be careful with array indexing - handle both empty arrays and different dimensions
                if len(intervals) > 0:
                    # Handle different array shapes
                    if intervals.ndim == 1 and len(intervals) >= 2:
                        # Single interval as a 1D array
                        start_idx, end_idx = int(intervals[0]), int(intervals[1])
                        segmentation[start_idx:end_idx+1] = 1
                    elif intervals.ndim == 2:
                        # Multiple intervals as a 2D array
                        for interval in intervals:
                            if len(interval) >= 2:
                                start_idx, end_idx = int(interval[0]), int(interval[1])
                                segmentation[start_idx:end_idx+1] = 1
                
                try:
                    # Use absolute path for visualization file to ensure it's saved in the current directory
                    abs_vis_path = os.path.abspath(visualization_file)
                    
                    # Always create visualization, even with no spindles
                    visualize_spindles(data, intervals, segmentation, output_path=abs_vis_path)
                    
                    # Print both relative (user-friendly) and absolute paths
                    rel_path = os.path.relpath(abs_vis_path)
                    console.print(f"[bold green]Visualization saved to:[/bold green] {rel_path}")
                    console.print(f"[dim]Full path: {abs_vis_path}[/dim]")
                except Exception as e:
                    console.print(f"[bold yellow]Warning:[/bold yellow] Visualization failed: {str(e)}")
            
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
        
        # Show usage examples at the end
        console.print("\n[bold]Usage examples:[/bold]")
        console.print("  spindle-detect detect path/to/data.txt --model-type eeg")
        console.print("  spindle-detect example --data-type eeg")

    @app.command("example")
    def run_example(
        data_type: str = typer.Option("eeg", help="Type of example data to use: 'eeg' or 'ieeg'"),
        visualization_file: Optional[Path] = typer.Option(None, help="Path to save visualization PDF (default: auto-generated)"),
        visualize: bool = typer.Option(True, "--visualize/--no-visualize", help="Visualize the detection results"),
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
                results = detect(data, model_type=data_type)
            except Exception as e:
                console.print(f"[bold red]Error during detection:[/bold red] {str(e)}")
                raise typer.Exit(code=1)
            
            # Output results
            intervals = results["detection_intervals"]
            console.print(f"[bold green]Found {len(intervals)} spindle(s)[/bold green]")
            
            # Print results to console
            if len(intervals) > 0:
                console.print("Spindle intervals (seconds):")
                # Print each interval - safely handle both [start, end] and [start, end, confidence] formats
                for i, interval in enumerate(intervals):
                    if len(interval) >= 2:  # Make sure we have at least start and end
                        start, end = interval[0], interval[1]
                        console.print(f"  {i+1}. {start:.2f} - {end:.2f} (duration: {end-start:.2f}s)")
            
            # Visualize if requested
            if visualize:
                console.print("Generating visualization...")
                
                # Auto-generate visualization filename if not provided
                if visualization_file is None:
                    visualization_file = Path(generate_output_filename(f"spindles_example_{data_type}"))
                
                # Create a simple segmentation mask for visualization
                # (1 for detected spindle segments, 0 elsewhere)
                segmentation = np.zeros(len(data))
                
                # Be careful with array indexing - handle both empty arrays and different dimensions
                if len(intervals) > 0:
                    # Handle different array shapes
                    if intervals.ndim == 1 and len(intervals) >= 2:
                        # Single interval as a 1D array
                        start_idx, end_idx = int(intervals[0]), int(intervals[1])
                        segmentation[start_idx:end_idx+1] = 1
                    elif intervals.ndim == 2:
                        # Multiple intervals as a 2D array
                        for interval in intervals:
                            if len(interval) >= 2:
                                start_idx, end_idx = int(interval[0]), int(interval[1])
                                segmentation[start_idx:end_idx+1] = 1
                
                try:
                    # Use absolute path for visualization file to ensure it's saved in the current directory
                    abs_vis_path = os.path.abspath(visualization_file)
                    
                    # Always create visualization, even with no spindles
                    visualize_spindles(data, intervals, segmentation, output_path=abs_vis_path)
                    
                    # Print both relative (user-friendly) and absolute paths
                    rel_path = os.path.relpath(abs_vis_path)
                    console.print(f"[bold green]Visualization saved to:[/bold green] {rel_path}")
                    console.print(f"[dim]Full path: {abs_vis_path}[/dim]")
                except Exception as e:
                    console.print(f"[bold yellow]Warning:[/bold yellow] Visualization failed: {str(e)}")
            
        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {str(e)}")
            raise typer.Exit(code=1)

    def main():
        """Entry point for the CLI."""
        app()

if __name__ == "__main__":
    main()
