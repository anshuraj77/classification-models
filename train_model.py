"""Train, evaluate, and save the project's implementation-based classifiers."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
from model_implementation.decision_tree_classifier import DecisionTreeModel
from model_implementation.gaussian_naive_bayes_classifier import GaussianNaiveBayesModel
from model_implementation.k_nearest_neighbor_classifier import KNearestNeighborModel
from model_implementation.logistic_regression import LogisticRegressionModel
from model_implementation.service.data_service import (
    FittedModelArtifact,
    TARGET_COLUMN,
)
from model_implementation.support_vector_machine import SupportVectorMachineModel

ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = ROOT_DIR / "data" / "bedtime_screentime_sleep_debt.csv"
DEFAULT_MODEL_DIR = ROOT_DIR / "model"


def get_model_specs() -> dict[str, type]:
    """Return the implementation class for each supported model."""
    return {
        "decision_tree": DecisionTreeModel,
        "gaussian_naive_bayes": GaussianNaiveBayesModel,
        "k_nearest_neighbors": KNearestNeighborModel,
        "logistic_regression": LogisticRegressionModel,
        "support_vector_machine": SupportVectorMachineModel,
    }


def train_and_save(data_path: Path, model_dir: Path, model_name: str = "all") -> None:
    if not data_path.is_file():
        raise FileNotFoundError(f"Training CSV not found: {data_path}")

    model_dir.mkdir(parents=True, exist_ok=True)
    model_specs = get_model_specs()
    if model_name != "all" and model_name not in model_specs:
        raise ValueError(f"Unknown model `{model_name}`.")
    selected_models = (
        model_specs.items()
        if model_name == "all"
        else [(model_name, model_specs[model_name])]
    )
    for selected_name, model_class in selected_models:
        model = model_class()
        model.load_data(str(data_path))
        model.fit()
        metrics = model.evaluate()
        if model.data is None:
            raise RuntimeError(f"{selected_name} did not load its training data.")
        artifact = {
            "model": FittedModelArtifact(
                model.model,
                model.data["feature_names"],
                model.data["numeric_features"],
                model.data["categorical_features"],
                model.data["categorical_encoder"],
                model.data["scaler"],
            ),
            "feature_names": model.data["feature_names"],
            "target_column": TARGET_COLUMN,
            "label_encoder": model.data["label_encoder"],
        }
        artifact_path = model_dir / f"{selected_name}.pkl"
        joblib.dump(artifact, artifact_path)
        print(f"{selected_name} evaluation: {metrics}")
        print(f"Saved {selected_name} to {artifact_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train and evaluate classification models using their implementations "
            "from model_implementation, then save the fitted estimators."
        )
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help=f"Training CSV path (default: {DEFAULT_DATA_PATH})",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help=f"Directory for saved model artifacts (default: {DEFAULT_MODEL_DIR})",
    )
    parser.add_argument(
        "--model",
        choices=["all", *get_model_specs().keys()],
        default="all",
        help="Model to train (default: all)",
    )
    arguments = parser.parse_args()
    train_and_save(arguments.data, arguments.model_dir, arguments.model)


if __name__ == "__main__":
    main()
