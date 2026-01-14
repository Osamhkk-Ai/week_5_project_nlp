import typer
from commands.generate import generate_command

app = typer.Typer()

@app.command()
def generate(
    topic: str = typer.Option("customer reviews",help="Subject/domain of the generated texts"),
    count: int = typer.Option(..., help="Number of rows to generate"),
    classes: list[str] = typer.Option(...,"--class", help="Repeat this option to add multiple classes. Example: --class positive --class negative"),
    api_key: str = typer.Option(..., help="Your Gemini API key (free tier)"),
):
    """
    Generate synthetic Arabic classification data using Gemini.
    Output is saved automatically to outputs/output_data/synthetic.csv
    """
    generate_command(
        topic=topic,
        count=count,
        classes=classes,
        api_key=api_key,
    )



    

if __name__ == "__main__":
    app()
