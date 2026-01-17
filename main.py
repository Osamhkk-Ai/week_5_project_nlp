import typer
from commands.generate import generate_app
from commands.eda import eda_app
from commands.preprocessing import preprocess_app
from commands.embedding import embedding_app

app = typer.Typer()

app.add_typer(generate_app, name="generate")
app.add_typer(eda_app, name="eda")
app.add_typer(preprocess_app, name="text")
app.add_typer(embedding_app, name="embd")



    

if __name__ == "__main__":
    app()
