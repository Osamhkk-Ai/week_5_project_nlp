import typer
from commands.generate import generate_app
from commands.eda import eda_app
from commands.preprocessing import preprocess_app
from commands.embedding import embedding_app
from commands.training import training_app

app = typer.Typer()

app.add_typer(generate_app, name="generate")
app.add_typer(eda_app, name="eda")
app.add_typer(preprocess_app, name="text")
app.add_typer(embedding_app, name="embd")
app.add_typer(training_app, name="model")



    

if __name__ == "__main__":
    app()
