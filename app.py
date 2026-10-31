from __future__ import annotations

from io import StringIO
import pickle
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)


ROOT_DIR = Path(__file__).resolve().parent
MODEL_DIR = ROOT_DIR / "model"
DEFAULT_TARGET_COLUMN = "sleep_debt_category"
MAX_UPLOAD_BYTES = 50 * 1024 * 1024


@st.cache_resource
def load_artifact(path: str, modified_time: float) -> Any:
    """Load a trusted, locally stored model artifact."""
    del modified_time  # Included in the cache key so replacing an artifact reloads it.
    return joblib.load(path)


def discover_models() -> list[Path]:
    if not MODEL_DIR.is_dir():
        return []
    return sorted(
        [*MODEL_DIR.glob("*.pkl"), *MODEL_DIR.glob("*.joblib")],
        key=lambda path: path.name.lower(),
    )


def unpack_artifact(artifact: Any) -> tuple[Any, dict[str, Any]]:
    if isinstance(artifact, dict) and "model" in artifact:
        metadata = {key: value for key, value in artifact.items() if key != "model"}
        return artifact["model"], metadata
    return artifact, {}


def get_feature_names(model: Any, metadata: dict[str, Any]) -> list[str] | None:
    names = metadata.get("feature_names")
    if names is None:
        names = getattr(model, "feature_names_in_", None)
    if names is None:
        return None
    if not isinstance(names, (list, tuple, np.ndarray, pd.Index)):
        raise ValueError("Model feature names must be a list or array of column names.")
    return [str(name) for name in names]


def evaluate_model(
    model: Any,
    metadata: dict[str, Any],
    dataset: pd.DataFrame,
) -> tuple[dict[str, float], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    feature_names = get_feature_names(model, metadata)
    if not feature_names:
        raise ValueError(
            "The model artifact does not define feature names. Save it with "
            "a `feature_names` list or fit it using a pandas DataFrame."
        )

    target_column = metadata.get("target_column", DEFAULT_TARGET_COLUMN)
    if target_column not in dataset.columns:
        raise ValueError(f"CSV is missing the target column `{target_column}`.")

    missing_features = [name for name in feature_names if name not in dataset.columns]
    if missing_features:
        raise ValueError(f"CSV is missing required feature(s): {', '.join(missing_features)}.")

    selected = dataset.loc[:, feature_names]
    if selected.isnull().any().any():
        null_columns = selected.columns[selected.isnull().any()].tolist()
        raise ValueError(f"Feature values are missing in: {', '.join(null_columns)}.")

    actual_labels = dataset[target_column]
    if actual_labels.isnull().any():
        raise ValueError(f"The target column `{target_column}` contains missing values.")
    if actual_labels.empty:
        raise ValueError("The uploaded CSV contains no test rows.")

    if "label_encoder" in metadata:
        encoder = metadata["label_encoder"]
        y_true = encoder.transform(actual_labels)
        class_labels = np.arange(len(encoder.classes_))
        class_names = [str(label) for label in encoder.classes_]
    else:
        y_true = actual_labels.to_numpy()
        class_labels = None
        class_names = None

    if hasattr(model, "feature_names_in_") or hasattr(model, "named_steps"):
        features = selected
    else:
        features = selected.to_numpy()

    predictions = np.asarray(model.predict(features))
    if predictions.ndim != 1 or len(predictions) != len(y_true):
        raise ValueError("The model returned predictions in an unexpected shape.")
    y_pred = predictions

    if class_labels is None:
        class_labels = pd.unique(np.concatenate([np.asarray(y_true), y_pred])).tolist()
        class_names = [str(label) for label in class_labels]

    predicted_labels = (
        metadata["label_encoder"].inverse_transform(y_pred)
        if "label_encoder" in metadata
        else y_pred
    )
    prediction_table = dataset.copy()
    prediction_table[f"Predicted {target_column}"] = predicted_labels

    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Weighted precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "Weighted recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "Weighted F1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "Matthews correlation": matthews_corrcoef(y_true, y_pred),
    }

    report = classification_report(
        y_true,
        y_pred,
        labels=class_labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    report_table = pd.DataFrame(report).transpose()
    matrix = confusion_matrix(y_true, y_pred, labels=class_labels)
    confusion_table = pd.DataFrame(
        matrix,
        index=pd.Index(class_names, name="Actual"),
        columns=pd.Index(class_names, name="Predicted"),
    )
    return metrics, confusion_table, report_table, prediction_table


def main() -> None:
    st.set_page_config(page_title="Classification Model Evaluation", page_icon="📊", layout="wide")
    st.title("Classification Model Evaluation")
    st.write(
        "Upload a labeled test CSV to evaluate saved models. The app only performs "
        "inference; it does not train or modify models."
    )

    model_paths = discover_models()
    if not model_paths:
        st.info(
            "No saved models found. Add trusted `.pkl` or `.joblib` model artifacts "
            "to the `model/` directory, then deploy the app."
        )

    uploaded_file = st.file_uploader(
        "Upload test data (CSV)",
        type=["csv"],
        help="Include every model input feature and the target column. Upload test data only.",
    )
    if uploaded_file is None:
        return
    if isinstance(uploaded_file, list):
        st.error("Upload one CSV file at a time.")
        return
    if uploaded_file.size > MAX_UPLOAD_BYTES:
        st.error("The CSV exceeds the 50 MB upload limit.")
        return
    if not model_paths:
        return

    try:
        csv_text = uploaded_file.getvalue().decode("utf-8-sig")
        dataset = pd.read_csv(StringIO(csv_text))
    except (UnicodeDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as error:
        st.error(f"Could not read the uploaded CSV: {error}")
        return

    dataset.columns = pd.Index([str(column).strip() for column in dataset.columns])
    if dataset.columns.duplicated().any():
        st.error("The CSV contains duplicate column names.")
        return
    if dataset.empty:
        st.error("The uploaded CSV contains no test rows.")
        return

    st.caption(f"Loaded {len(dataset):,} test rows.")
    comparison: list[dict[str, Any]] = []
    details: dict[
        str, tuple[dict[str, float], pd.DataFrame, pd.DataFrame, pd.DataFrame]
    ] = {}
    failures: list[tuple[str, str]] = []

    for path in model_paths:
        try:
            artifact = load_artifact(str(path), path.stat().st_mtime)
            model, metadata = unpack_artifact(artifact)
            result = evaluate_model(model, metadata, dataset)
            metrics, confusion_table, report_table, _ = result
            details[path.name] = result
            comparison.append({"Model": path.stem, **metrics})
        except (
            OSError,
            EOFError,
            pickle.UnpicklingError,
            ImportError,
            ValueError,
            TypeError,
            AttributeError,
            KeyError,
        ) as error:
            failures.append((path.name, str(error)))

    if comparison:
        st.subheader("Model comparison")
        st.dataframe(
            pd.DataFrame(comparison).set_index("Model").style.format("{:.4f}"),
            use_container_width=True,
        )
        selected_model = st.selectbox("Inspect model results", list(details))
        metrics, confusion_table, report_table, prediction_table = details[
            selected_model
        ]

        st.subheader(f"{Path(selected_model).stem} details")
        metric_columns = st.columns(len(metrics))
        for column, (name, value) in zip(metric_columns, metrics.items()):
            column.metric(name, f"{value:.4f}")

        st.subheader("Predictions")
        st.caption(
            "Uploaded rows with the selected model's predicted target. "
            "Scroll within the table to inspect all rows."
        )
        st.dataframe(
            prediction_table,
            height=400,
            use_container_width=True,
            hide_index=True,
        )

        confusion_tab, report_tab = st.tabs(["Confusion matrix", "Classification report"])
        with confusion_tab:
            st.dataframe(confusion_table, use_container_width=True)
        with report_tab:
            st.dataframe(report_table.style.format("{:.4f}"), use_container_width=True)

    if failures:
        st.subheader("Models that could not be evaluated")
        for model_name, reason in failures:
            st.error(f"{model_name}: {reason}")

    st.caption(
        "Model artifacts are loaded with joblib/pickle. Only deploy artifacts from "
        "sources you trust."
    )


if __name__ == "__main__":
    main()
