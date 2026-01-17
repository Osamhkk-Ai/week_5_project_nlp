import typer
from typing import List, Optional
import pickle
from pathlib import Path

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from utils.data_handler import load_data, split_data
from utils.metrics import compute_classification_metrics

training_app = typer.Typer()

def build_model(name: str, params: dict | None = None):
    params = params or {}
    name = name.lower()

    if name == "knn":
        return KNeighborsClassifier(**params)

    if name in ["lr", "logistic", "regression"]:
        return LogisticRegression(max_iter=1000, **params)

    if name in ["rf", "random", "forest"]:
        return RandomForestClassifier(**params)

    raise ValueError(f"Unsupported model: {name}")


def parse_models(models: List[str]):
    """
    Returns list of (model_name, params)
    """

    if "all" in models:
        if len(models) > 1:
            print("⚠️  'all' selected → hyperparameters will be ignored.")
        return [
            ("knn", {}),
            ("logistic", {}),
            ("forest", {}),
        ]

    parsed = []

    for item in models:
        if ":" in item:
            name, param_str = item.split(":", 1)
            params = {}
            for p in param_str.split(","):
                k, v = p.split("=")
                try:
                    v = int(v)
                except ValueError:
                    try:
                        v = float(v)
                    except ValueError:
                        pass
                params[k] = v
        else:
            name = item
            params = {}

        parsed.append((name, params))

    return parsed

from datetime import datetime


def write_training_report(
    results: dict,
    num_samples: int,
    num_features: int,
    test_size: float,
    output_dir: str = "outputs/reports",
):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    report_path = Path(output_dir)
    report_path.mkdir(parents=True, exist_ok=True)

    report_file = report_path / f"training_report_{timestamp}.md"

    best_model = max(results, key=lambda k: results[k]["f1"])

    with open(report_file, "w", encoding="utf-8") as f:
        f.write(f"# Training Report - {timestamp}\n\n")

        f.write("## Dataset Info\n")
        f.write(f"- Total samples: {num_samples}\n")
        f.write(f"- Test size: {int(test_size * 100)}%\n")
        f.write(f"- Features: {num_features}\n\n")

        f.write("## Model Performance\n\n")

        for name, m in results.items():
            f.write(f"### {name.upper()}\n")
            f.write(f"- Accuracy:  {m['accuracy']:.3f}\n")
            f.write(f"- Precision: {m['precision']:.3f}\n")
            f.write(f"- Recall:    {m['recall']:.3f}\n")
            f.write(f"- F1-score:  {m['f1']:.3f}\n\n")

        f.write(f"## Best Model ⭐\n")
        f.write(f"**{best_model.upper()}**\n")

    return report_file



@training_app.command()
def train(
    data_path: str = typer.Option(..., help="CSV or PKL data path"),
    models: List[str] = typer.Option(..., help="Model names or 'all'"),
    output_col: str = typer.Option(..., help="Label column"),
    input_col: Optional[str] = typer.Option(None, help="Input column (for CSV)"),
    test_size: float = typer.Option(0.2),
    save_model: Optional[str] = typer.Option(None, help="Save best model name"),
):
    X, y = load_data(data_path, input_col, output_col)
    X_train, X_test, y_train, y_test = split_data(X, y, test_size)

    parsed_models = parse_models(models)

    results = {}
    trained_models = {}

    for name, params in parsed_models:
        model = build_model(name, params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        metrics = compute_classification_metrics(y_test, y_pred)
        results[name] = metrics
        trained_models[name] = model

    print("\n Training Results")
    print("-" * 30)
    for name, m in results.items():
        print(f"\n### {name.upper()}")
        print(f"Accuracy : {m['accuracy']:.3f}")
        print(f"Precision: {m['precision']:.3f}")
        print(f"Recall   : {m['recall']:.3f}")
        print(f"F1-score : {m['f1']:.3f}")

    if save_model:
        best = max(results, key=lambda k: results[k]["f1"])
        Path("outputs/models").mkdir(parents=True, exist_ok=True)
        with open(f"outputs/models/{save_model}", "wb") as f:
            pickle.dump(trained_models[best], f)

        print(f"\n Best model saved: outputs/models/{save_model}")

        report_file = write_training_report(
        results=results,
        num_samples=len(X),
        num_features=X.shape[1],
        test_size=test_size,
        )

        print(f"\n📝 Training report saved to: {report_file}")

