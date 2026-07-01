import sys

import click
import requests

SERVER_URL = "http://localhost:8421"


@click.command()
@click.argument("update_type", type=click.Choice(["name", "type"], case_sensitive=False))
@click.argument("global_name")
@click.argument("new_name", required=False)
@click.argument("new_type", required=False, type=click.Path())
def update_global(update_type: str, global_name: str, new_name: str | None, new_type: str | None) -> None:
    """
    A tool for updating global variables inside LLVM bitcode files.
    You can change the name of a global variable or update its type definition using C syntax.
    """
    try:
        if update_type == "name":
            if not new_name:
                click.echo("Error: new_name is required for 'name' update.", err=True)
                sys.exit(1)
            resp = requests.post(
                f"{SERVER_URL}/set_global_name", json={"global_name": global_name, "new_name": new_name}
            )
            if resp.ok:
                click.echo(resp.json())
            else:
                click.echo(f"Error: {resp.text}", err=True)
                sys.exit(1)
        elif update_type == "type":
            if not new_type:
                click.echo("Error: new_type are required for 'type' update.", err=True)
                sys.exit(1)
            with open(new_type) as f:
                type_content = f.read()
            resp = requests.post(
                f"{SERVER_URL}/set_global_type", json={"global_name": global_name, "new_type": type_content}
            )
            if resp.ok:
                click.echo(resp.json())
            else:
                click.echo(f"Error: {resp.text}", err=True)
                sys.exit(1)
        else:
            click.echo(f"Unknown update_type: {update_type}", err=True)
            sys.exit(1)
    except requests.ConnectionError:
        click.echo("Could not connect to the binary server. Is it running?", err=True)
        sys.exit(1)


if __name__ == "__main__":
    update_global()
