import json
import sys
from pathlib import Path

import typer
from jinja2 import Environment, FileSystemLoader

from swereview.model import Trajectory

app = typer.Typer()


def render_trajectory(traj_path: Path, output_path: Path) -> None:
    """Render trajectory to HTML file"""
    # Load trajectory
    with open(traj_path, encoding="utf-8") as f:
        data = json.load(f)
    trajectory = Trajectory.load_dict(data)

    # Setup jinja2 environment
    templates_dir = Path(__file__).parent.parent / "templates"
    env = Environment(loader=FileSystemLoader(templates_dir))
    template = env.get_template("trajectory.html")

    # force reset data
    trajectory.messages = trajectory.messages
    trajectory.history = trajectory.history
    trajectory.trajectory = []

    # Render HTML
    html = template.render(trajectory=trajectory, file_name=traj_path.stem)

    # Write to file
    output_path.write_text(html)


@app.command(name="plain", help="render trajectory.json to human readable plain text")
def render(
    traj_path: Path = typer.Argument(..., help="Path to Trajectory JSON file"),
    output_path: Path = typer.Option(
        None,
        "-o",
        "--output",
        help="Output HTML file path, defaults to same name with html extension",
    ),
):
    # Check input file
    if not traj_path.exists():
        typer.echo(f"Error: File {traj_path} does not exist")
        sys.exit(1)

    # Load trajectory
    with open(traj_path, encoding="utf-8") as f:
        data = json.load(f)
    trajectory = Trajectory.load_dict(data)

    lines = []
    last_role = None
    for idx, msg in enumerate(trajectory.history):
        role = msg.role
        if role != last_role:
            banner = "===" * 10
            header = f"{banner} step {idx} === {role} {banner}"
            lines.append(header)
            last_role = role
        content = msg.content
        lines.append(f"{content}")

    plain_text = "\n".join(lines)

    # Output path
    if output_path:
        output_path = traj_path.with_suffix(".txt")
        output_path.write_text(plain_text, encoding="utf-8")
        typer.echo(f"Plain text file generated: {output_path}")
    else:
        print(plain_text)


@app.command(name="html", help="render trajectory.json to html page")
def render(
    traj_path: Path = typer.Argument(..., help="Path to Trajectory JSON file"),
    output_path: Path = typer.Option(
        None,
        "-o",
        "--output",
        help="Output HTML file path, defaults to same name with html extension",
    ),
    open_browser: bool = typer.Option(True, "-b", "---browse", help="Whether to automatically open browser to view"),
):
    """Render trajectory.json to HTML and open it"""
    # Check input file
    if not traj_path.exists():
        typer.echo(f"Error: File {traj_path} does not exist")
        sys.exit(1)

    # Set output path
    if output_path is None:
        output_path = traj_path.with_suffix(".html")

    # Render HTML
    render_trajectory(traj_path, output_path)
    typer.echo(f"HTML file generated: {output_path}")

    # Open browser
    if open_browser:
        import webbrowser

        webbrowser.open(f"file://{output_path.absolute()}")


if __name__ == "__main__":
    app()
