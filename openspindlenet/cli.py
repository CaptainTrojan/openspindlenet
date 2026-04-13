"""Command-line interface for OpenSpindleNet using Typer."""

from __future__ import annotations

import csv
import datetime
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Optional


CLI_AVAILABLE = True
MISSING_PACKAGES = []
for package in ["typer", "rich", "tqdm"]:
    if importlib.util.find_spec(package) is None:
        CLI_AVAILABLE = False
        MISSING_PACKAGES.append(package)


if not CLI_AVAILABLE:
    def main():
        print("Error: OpenSpindleNet CLI dependencies are not installed.")
        print(f"Missing packages: {', '.join(MISSING_PACKAGES)}")
        print("\nTo use the CLI features, install the package with CLI extras:")
        print("    pip install openspindlenet[cli]")
        sys.exit(1)
else:
    import inspect
    import numpy as np
    import typer
    import click
    from rich.console import Console
    from tqdm import tqdm

    # Typer/Click compatibility:
    # Some environments mix Typer with Click where make_metavar() requires ctx,
    # while Typer rich help may call it without ctx.
    _make_metavar_sig = inspect.signature(click.core.Parameter.make_metavar)
    if "ctx" in _make_metavar_sig.parameters:
        _orig_make_metavar = click.core.Parameter.make_metavar

        def _compat_make_metavar(self, ctx=None):
            if ctx is None:
                ctx = click.get_current_context(silent=True)
            if ctx is None:
                ctx = click.Context(click.Command(name="openspindlenet"))
            return _orig_make_metavar(self, ctx)

        click.core.Parameter.make_metavar = _compat_make_metavar

    from . import detect
    from . import get_example_data_path, get_model_path
    from .evaluator import Evaluator
    from .visualization import visualize_spindles

    app = typer.Typer(
        help=(
            "OpenSpindleNet: Detect and evaluate sleep spindles in EEG/iEEG data\n\n"
            "Examples:\n"
            "  openspindlenet detect path/to/signal.txt --model-type eeg --visualize\n"
            "  openspindlenet eval path/to/signal.txt path/to/labels.txt --visualize"
        )
    )
    console = Console()

    def _timestamp() -> str:
        return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    def _auto_output_path(prefix: str, suffix: str, stem: str) -> Path:
        return Path(f"{prefix}_{stem}_{_timestamp()}.{suffix}")

    def _load_signal(path: Path) -> np.ndarray:
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        data = np.loadtxt(path, dtype=np.float32)
        if data.ndim != 1:
            data = data.reshape(-1)
        return data

    def _load_labels(path: Path) -> np.ndarray:
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        labels = np.loadtxt(path, dtype=np.float32)
        if labels.ndim != 1:
            labels = labels.reshape(-1)
        return labels

    def _pred_segmentation(results: dict, seq_len: int, threshold: float) -> np.ndarray:
        seg = results.get("segmentation", np.zeros((seq_len, 1), dtype=np.float32))
        seg = np.asarray(seg)
        if seg.ndim == 2:
            seg = seg[:, 0]
        return (seg > threshold).astype(np.float32)

    def _mean_std(values: list[float]) -> tuple[float, float]:
        if not values:
            return 0.0, 0.0
        arr = np.asarray(values, dtype=np.float64)
        return float(arr.mean()), float(arr.std(ddof=1)) if len(arr) > 1 else 0.0

    def _prf_from_counts(tp: int, fp: int, fn: int) -> dict:
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        return {
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }

    def _collect_eval_pairs(signal_dir: Path, labels_dir: Path) -> list[tuple[Path, Path, str]]:
        signal_files = sorted(signal_dir.glob("*.txt"))
        if not signal_files:
            raise ValueError(f"No .txt files found in signal directory: {signal_dir}")

        pairs: list[tuple[Path, Path, str]] = []
        missing_labels: list[str] = []
        for signal_path in signal_files:
            label_path = labels_dir / signal_path.name
            if not label_path.exists():
                missing_labels.append(signal_path.name)
                continue
            pairs.append((signal_path, label_path, signal_path.stem))

        if missing_labels:
            raise ValueError(
                "Missing label files for signals: " + ", ".join(missing_labels[:10])
                + (" ..." if len(missing_labels) > 10 else "")
            )

        return pairs

    def _build_dir_summary(per_file_rows: list[dict], model_type: str, thresholds: dict) -> dict:
        det_p = [float(r["det_precision"]) for r in per_file_rows]
        det_r = [float(r["det_recall"]) for r in per_file_rows]
        det_f = [float(r["det_f1"]) for r in per_file_rows]
        seg_p = [float(r["seg_precision"]) for r in per_file_rows]
        seg_r = [float(r["seg_recall"]) for r in per_file_rows]
        seg_f = [float(r["seg_f1"]) for r in per_file_rows]

        det_tp = int(sum(int(r["det_tp"]) for r in per_file_rows))
        det_fp = int(sum(int(r["det_fp"]) for r in per_file_rows))
        det_fn = int(sum(int(r["det_fn"]) for r in per_file_rows))
        seg_tp = int(sum(int(r["seg_tp"]) for r in per_file_rows))
        seg_fp = int(sum(int(r["seg_fp"]) for r in per_file_rows))
        seg_fn = int(sum(int(r["seg_fn"]) for r in per_file_rows))

        det_macro_p = _mean_std(det_p)
        det_macro_r = _mean_std(det_r)
        det_macro_f = _mean_std(det_f)
        seg_macro_p = _mean_std(seg_p)
        seg_macro_r = _mean_std(seg_r)
        seg_macro_f = _mean_std(seg_f)

        return {
            "num_files": int(len(per_file_rows)),
            "model_type": model_type,
            "thresholds": thresholds,
            "detection": {
                "macro": {
                    "precision_mean": det_macro_p[0],
                    "precision_std": det_macro_p[1],
                    "recall_mean": det_macro_r[0],
                    "recall_std": det_macro_r[1],
                    "f1_mean": det_macro_f[0],
                    "f1_std": det_macro_f[1],
                },
                "micro": _prf_from_counts(det_tp, det_fp, det_fn),
                "counts": {"tp": det_tp, "fp": det_fp, "fn": det_fn},
            },
            "segmentation": {
                "macro": {
                    "precision_mean": seg_macro_p[0],
                    "precision_std": seg_macro_p[1],
                    "recall_mean": seg_macro_r[0],
                    "recall_std": seg_macro_r[1],
                    "f1_mean": seg_macro_f[0],
                    "f1_std": seg_macro_f[1],
                },
                "micro": _prf_from_counts(seg_tp, seg_fp, seg_fn),
                "counts": {"tp": seg_tp, "fp": seg_fp, "fn": seg_fn},
            },
        }

    @app.command("detect")
    def detect_cmd(
        input_file: Path = typer.Argument(..., help="Path to input EEG/iEEG TXT file"),
        model_type: str = typer.Option("eeg", help="Model type: eeg or ieeg"),
        output_file: Optional[Path] = typer.Option(None, help="Optional output TXT with detection intervals"),
        visualization_file: Optional[Path] = typer.Option(None, help="Optional output PDF path"),
        visualize: bool = typer.Option(False, "--visualize/--no-visualize", help="Generate PDF visualization"),
        confidence_threshold: float = typer.Option(0.5, help="Confidence threshold for detections"),
        nms_iou_threshold: float = typer.Option(0.3, help="NMS IoU threshold"),
    ):
        """Detect spindles in a single signal file."""
        try:
            data = _load_signal(input_file)
            results = detect(
                data,
                model_type=model_type,
                confidence_threshold=confidence_threshold,
                nms_iou_threshold=nms_iou_threshold,
            )
            intervals = np.asarray(results.get("detection_intervals", []))
            console.print(f"[bold green]Detected {len(intervals)} spindle(s).[/bold green]")

            if output_file:
                header = "start end confidence"
                np.savetxt(output_file, intervals, fmt="%.6f", header=header)
                console.print(f"Saved intervals to [bold]{output_file}[/bold]")

            for i, interval in enumerate(intervals):
                start, end = float(interval[0]), float(interval[1])
                conf = float(interval[2]) if len(interval) >= 3 else float("nan")
                console.print(f"  {i + 1}. start={start:.1f}, end={end:.1f}, conf={conf:.3f}")

            if visualize:
                out_pdf = visualization_file or _auto_output_path("detect", "pdf", input_file.stem)
                pred_seg = _pred_segmentation(results, len(data), threshold=0.5)
                visualize_spindles(
                    signal=data,
                    intervals=intervals,
                    segmentation=pred_seg,
                    output_path=str(out_pdf),
                )
                console.print(f"Saved visualization to [bold]{out_pdf}[/bold]")
        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {e}")
            raise typer.Exit(code=1)

    @app.command("eval")
    def eval_cmd(
        signal_file: Path = typer.Argument(..., help="Path to signal TXT file"),
        labels_file: Path = typer.Argument(..., help="Path to labels TXT file"),
        model_type: str = typer.Option("eeg", help="Model type: eeg or ieeg"),
        output_dir: Optional[Path] = typer.Option(None, help="Output directory for directory-mode evaluation"),
        output_json: Optional[Path] = typer.Option(None, help="Optional output JSON path"),
        output_csv: Optional[Path] = typer.Option(None, help="Optional output CSV path"),
        visualization_file: Optional[Path] = typer.Option(None, help="Optional output PDF path"),
        visualize: bool = typer.Option(False, "--visualize/--no-visualize", help="Generate PDF with predictions, labels, and metrics"),
        confidence_threshold: float = typer.Option(0.5, help="Detection confidence threshold"),
        nms_iou_threshold: float = typer.Option(0.3, help="Detection NMS IoU threshold"),
        match_iou_threshold: float = typer.Option(0.3, help="Interval match IoU threshold"),
        segmentation_threshold: float = typer.Option(0.5, help="Segmentation threshold"),
    ):
        """Evaluate predictions against label segmentation for one signal."""
        try:
            if signal_file.is_dir() or labels_file.is_dir():
                if not signal_file.is_dir() or not labels_file.is_dir():
                    raise ValueError("Directory-mode eval requires both SIGNAL and LABELS arguments to be directories")

                pairs = _collect_eval_pairs(signal_file, labels_file)
                out_dir = output_dir or Path(f"eval_dir_{signal_file.name}_{_timestamp()}")
                vis_dir = out_dir / "visualizations"
                out_dir.mkdir(parents=True, exist_ok=True)
                vis_dir.mkdir(parents=True, exist_ok=True)

                per_file_rows: list[dict] = []
                thresholds = {
                    "confidence_threshold": confidence_threshold,
                    "nms_iou_threshold": nms_iou_threshold,
                    "match_iou_threshold": match_iou_threshold,
                    "segmentation_threshold": segmentation_threshold,
                }

                console.print(f"[bold]Directory evaluation:[/bold] {len(pairs)} files")
                for signal_path, label_path, sample_id in tqdm(pairs, desc="Evaluating files", unit="file"):
                    signal = _load_signal(signal_path)
                    labels = _load_labels(label_path)
                    if len(signal) != len(labels):
                        raise ValueError(
                            f"Length mismatch in {sample_id}: signal={len(signal)}, labels={len(labels)}"
                        )

                    results = detect(
                        signal,
                        model_type=model_type,
                        labels=labels,
                        confidence_threshold=confidence_threshold,
                        nms_iou_threshold=nms_iou_threshold,
                        match_iou_threshold=match_iou_threshold,
                        segmentation_threshold=segmentation_threshold,
                    )

                    intervals = np.asarray(results.get("detection_intervals", []))
                    metrics = results.get("metrics", {})
                    det = metrics.get("detection", {})
                    seg = metrics.get("segmentation", {})

                    per_file_rows.append(
                        {
                            "sample_id": sample_id,
                            "signal_file": str(signal_path),
                            "labels_file": str(label_path),
                            "model_type": model_type,
                            "num_predicted_intervals": int(len(intervals)),
                            "det_tp": int(det.get("tp", 0)),
                            "det_fp": int(det.get("fp", 0)),
                            "det_fn": int(det.get("fn", 0)),
                            "det_precision": float(det.get("precision", 0.0)),
                            "det_recall": float(det.get("recall", 0.0)),
                            "det_f1": float(det.get("f1", 0.0)),
                            "seg_tp": int(seg.get("tp", 0)),
                            "seg_fp": int(seg.get("fp", 0)),
                            "seg_fn": int(seg.get("fn", 0)),
                            "seg_precision": float(seg.get("precision", 0.0)),
                            "seg_recall": float(seg.get("recall", 0.0)),
                            "seg_f1": float(seg.get("f1", 0.0)),
                        }
                    )

                    # Directory-mode always writes per-file visualizations into visualizations/.
                    out_pdf = vis_dir / f"{sample_id}.pdf"
                    pred_seg = _pred_segmentation(results, len(signal), threshold=segmentation_threshold)
                    true_seg = labels.astype(np.float32)
                    visualize_spindles(
                        signal=signal,
                        intervals=intervals,
                        segmentation=pred_seg,
                        output_path=str(out_pdf),
                        true_segmentation=true_seg,
                        metrics=metrics,
                        title=f"Evaluation: {sample_id}",
                        close_figure=True,
                    )

                summary = _build_dir_summary(per_file_rows, model_type=model_type, thresholds=thresholds)

                out_csv = out_dir / "per_file_results.csv"
                out_json = out_dir / "summary.json"

                with open(out_csv, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=list(per_file_rows[0].keys()))
                    writer.writeheader()
                    writer.writerows(per_file_rows)

                with open(out_json, "w", encoding="utf-8") as f:
                    json.dump(summary, f, indent=2)

                det_micro = summary["detection"]["micro"]
                seg_micro = summary["segmentation"]["micro"]
                console.print("[bold green]Directory evaluation complete.[/bold green]")
                console.print(
                    "Detection micro: "
                    f"P={det_micro['precision']:.4f} "
                    f"R={det_micro['recall']:.4f} "
                    f"F1={det_micro['f1']:.4f}"
                )
                console.print(
                    "Segmentation micro: "
                    f"P={seg_micro['precision']:.4f} "
                    f"R={seg_micro['recall']:.4f} "
                    f"F1={seg_micro['f1']:.4f}"
                )
                console.print(f"Saved per-file CSV to [bold]{out_csv}[/bold]")
                console.print(f"Saved summary JSON to [bold]{out_json}[/bold]")
                console.print(f"Saved per-file PDFs to [bold]{vis_dir}[/bold]")
                return

            signal = _load_signal(signal_file)
            labels = _load_labels(labels_file)
            if len(signal) != len(labels):
                raise ValueError(f"Signal length ({len(signal)}) and labels length ({len(labels)}) must match")

            results = detect(
                signal,
                model_type=model_type,
                labels=labels,
                confidence_threshold=confidence_threshold,
                nms_iou_threshold=nms_iou_threshold,
                match_iou_threshold=match_iou_threshold,
                segmentation_threshold=segmentation_threshold,
            )

            intervals = np.asarray(results.get("detection_intervals", []))
            metrics = results.get("metrics", {})
            det = metrics.get("detection", {})
            seg = metrics.get("segmentation", {})

            console.print("[bold green]Evaluation complete.[/bold green]")
            console.print(
                "Detection: "
                f"P={det.get('precision', 0.0):.4f} "
                f"R={det.get('recall', 0.0):.4f} "
                f"F1={det.get('f1', 0.0):.4f} "
                f"(TP={det.get('tp', 0)}, FP={det.get('fp', 0)}, FN={det.get('fn', 0)})"
            )
            console.print(
                "Segmentation: "
                f"P={seg.get('precision', 0.0):.4f} "
                f"R={seg.get('recall', 0.0):.4f} "
                f"F1={seg.get('f1', 0.0):.4f} "
                f"(TP={seg.get('tp', 0)}, FP={seg.get('fp', 0)}, FN={seg.get('fn', 0)})"
            )

            out_json = output_json or _auto_output_path("eval", "json", signal_file.stem)
            out_csv = output_csv or _auto_output_path("eval", "csv", signal_file.stem)

            json_payload = {
                "signal_file": str(signal_file),
                "labels_file": str(labels_file),
                "model_type": model_type,
                "thresholds": {
                    "confidence_threshold": confidence_threshold,
                    "nms_iou_threshold": nms_iou_threshold,
                    "match_iou_threshold": match_iou_threshold,
                    "segmentation_threshold": segmentation_threshold,
                },
                "num_predicted_intervals": int(len(intervals)),
                "metrics": metrics,
                "detection_intervals": intervals.tolist(),
            }
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(json_payload, f, indent=2)

            csv_row = {
                "signal_file": str(signal_file),
                "labels_file": str(labels_file),
                "model_type": model_type,
                "num_predicted_intervals": int(len(intervals)),
                "det_tp": int(det.get("tp", 0)),
                "det_fp": int(det.get("fp", 0)),
                "det_fn": int(det.get("fn", 0)),
                "det_precision": float(det.get("precision", 0.0)),
                "det_recall": float(det.get("recall", 0.0)),
                "det_f1": float(det.get("f1", 0.0)),
                "seg_tp": int(seg.get("tp", 0)),
                "seg_fp": int(seg.get("fp", 0)),
                "seg_fn": int(seg.get("fn", 0)),
                "seg_precision": float(seg.get("precision", 0.0)),
                "seg_recall": float(seg.get("recall", 0.0)),
                "seg_f1": float(seg.get("f1", 0.0)),
            }
            with open(out_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(csv_row.keys()))
                writer.writeheader()
                writer.writerow(csv_row)

            console.print(f"Saved JSON to [bold]{out_json}[/bold]")
            console.print(f"Saved CSV to [bold]{out_csv}[/bold]")

            if visualize:
                out_pdf = visualization_file or _auto_output_path("eval", "pdf", signal_file.stem)
                pred_seg = _pred_segmentation(results, len(signal), threshold=segmentation_threshold)
                true_seg = labels.astype(np.float32)
                visualize_spindles(
                    signal=signal,
                    intervals=intervals,
                    segmentation=pred_seg,
                    output_path=str(out_pdf),
                    true_segmentation=true_seg,
                    metrics=metrics,
                    title=f"Evaluation: {signal_file.name}",
                )
                console.print(f"Saved visualization to [bold]{out_pdf}[/bold]")
        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {e}")
            raise typer.Exit(code=1)

    @app.command("info")
    def info():
        """Display information about OpenSpindleNet models and examples."""
        console.print("[bold]OpenSpindleNet[/bold]")
        console.print("\n[bold]Available models:[/bold]")

        try:
            console.print(f"  - EEG: [green]{get_model_path('eeg')}[/green]")
        except Exception:
            console.print("  - EEG: [red]Not found[/red]")

        try:
            console.print(f"  - iEEG: [green]{get_model_path('ieeg')}[/green]")
        except Exception:
            console.print("  - iEEG: [red]Not found[/red]")

        console.print("\n[bold]Example data:[/bold]")
        try:
            console.print(f"  - EEG: [green]{get_example_data_path('eeg')}[/green]")
        except Exception:
            console.print("  - EEG: [red]Not found[/red]")

        try:
            console.print(f"  - iEEG: [green]{get_example_data_path('ieeg')}[/green]")
        except Exception:
            console.print("  - iEEG: [red]Not found[/red]")

        console.print("\n[bold]Usage examples:[/bold]")
        console.print("  openspindlenet detect path/to/signal.txt --model-type eeg")
        console.print("  openspindlenet eval path/to/signal.txt path/to/labels.txt --visualize")

    @app.command("example")
    def run_example(
        data_type: str = typer.Option("eeg", help="Example data type: eeg or ieeg"),
        visualization_file: Optional[Path] = typer.Option(None, help="Optional output PDF path"),
        visualize: bool = typer.Option(True, "--visualize/--no-visualize", help="Generate visualization"),
    ):
        """Run spindle detection on packaged example data."""
        try:
            data_path = Path(get_example_data_path(data_type))
            data = _load_signal(data_path)
            results = detect(data, model_type=data_type)
            intervals = np.asarray(results.get("detection_intervals", []))

            console.print(f"[bold green]Detected {len(intervals)} spindle(s) in {data_type} example.[/bold green]")
            for i, interval in enumerate(intervals):
                start, end = float(interval[0]), float(interval[1])
                conf = float(interval[2]) if len(interval) >= 3 else float("nan")
                console.print(f"  {i + 1}. start={start:.1f}, end={end:.1f}, conf={conf:.3f}")

            if visualize:
                out_pdf = visualization_file or _auto_output_path("example", "pdf", data_type)
                pred_seg = _pred_segmentation(results, len(data), threshold=0.5)
                visualize_spindles(
                    signal=data,
                    intervals=intervals,
                    segmentation=pred_seg,
                    output_path=str(out_pdf),
                )
                console.print(f"Saved visualization to [bold]{out_pdf}[/bold]")
        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {e}")
            raise typer.Exit(code=1)

    def main():
        app()

if __name__ == "__main__":
    main()
