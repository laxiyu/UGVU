"""UGVU CLI — python -m ugvu 入口点.

用法:
    python -m ugvu run -c configs/default.yaml
    python -m ugvu calibrate -c configs/default.yaml
    python -m ugvu robustness -c configs/default.yaml -p 50
"""

import logging
import click

from .pipeline import UGVUPipeline


def _setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


@click.group()
@click.version_option(version="1.0.0")
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging.")
def cli(verbose):
    """UGVU — Uncertainty-Guided Generative Vision Understanding.

    A framework for reliable dense prediction from black-box image generators.
    """
    _setup_logging(verbose)


@cli.command()
@click.option("--config", "-c", required=True, type=click.Path(exists=True), help="Path to YAML config file.")
@click.option("--task", "-t", default="semantic_segmentation", help="Task type.")
@click.option("--output", "-o", default="outputs/", help="Output directory.")
def run(config, task, output):
    """Run full UGVU pipeline end-to-end."""
    pipeline = UGVUPipeline(config, task=task, output_dir=output)
    metrics = pipeline.run()
    click.echo(f"\nPipeline complete. mIoU = {metrics.get('mIoU', 'N/A')}")


@cli.command()
@click.option("--config", "-c", required=True, type=click.Path(exists=True))
@click.option("--task", "-t", default="semantic_segmentation")
@click.option("--output", "-o", default=None, help="Output directory with existing predictions.")
def evaluate(config, task, output):
    """Run evaluation on pre-computed predictions."""
    pipeline = UGVUPipeline(config, task=task, output_dir=output)
    if not pipeline.load_results():
        click.echo("Error: No predictions found. Run 'ugvu run' first.", err=True)
        raise SystemExit(1)
    metrics = pipeline.evaluate()
    click.echo(f"\nEvaluation complete. mIoU = {metrics.get('mIoU', 'N/A')}")


@cli.command()
@click.option("--config", "-c", required=True, type=click.Path(exists=True))
@click.option("--task", "-t", default="semantic_segmentation")
@click.option("--num-prompts", "-p", default=50, help="Number of prompt variants.")
@click.option("--output", "-o", default=None, help="Output directory.")
def robustness(config, task, num_prompts, output):
    """Run GVU-Robust benchmark."""
    pipeline = UGVUPipeline(config, task=task, output_dir=output)
    result = pipeline.run_robustness(num_prompts=num_prompts)
    summary = result.get("summary", {})
    click.echo(f"\nRobustness benchmark complete.")
    click.echo(f"  PRS = {summary.get('prs', 'N/A')}")
    click.echo(f"  GRS = {summary.get('grs', 'N/A')}")
    click.echo(f"  MRS = {summary.get('mrs', 'N/A')}")


@cli.command()
@click.option("--config", "-c", required=True, type=click.Path(exists=True))
@click.option("--task", "-t", default="semantic_segmentation")
@click.option("--output", "-o", default=None, help="Output directory with existing predictions.")
@click.option("--run-first", is_flag=True, help="Run full pipeline before calibration.")
def calibrate(config, task, output, run_first):
    """Run calibration analysis."""
    pipeline = UGVUPipeline(config, task=task, output_dir=output)
    if run_first:
        pipeline.run()
    else:
        if not pipeline.load_results():
            click.echo("Error: No predictions found. Use --run-first or run 'ugvu run' first.", err=True)
            raise SystemExit(1)
    result = pipeline.run_calibration()
    click.echo(f"\nCalibration complete.")
    click.echo(f"  ECE = {result['ece']:.4f}")
    click.echo(f"  Spearman rho = {result['spearman_r']:.4f}")
    click.echo(f"  AUROC = {result['auroc']:.4f}")


if __name__ == "__main__":
    cli()
