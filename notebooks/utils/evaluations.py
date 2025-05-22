import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import Pipeline


def evaluate_model(
    model: ClassifierMixin,
    x_train: pd.DataFrame,
    x_val: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
) -> dict:
    """Evaluates a model's performance on validation data and returns metrics."""
    model.fit(x_train, y_train)
    y_pred = model.predict(x_val)
    y_proba = model.predict_proba(x_val)[:, 1]
    return {
        "accuracy": accuracy_score(y_val, y_pred),
        "precision": precision_score(y_val, y_pred),
        "recall": recall_score(y_val, y_pred),
        "f1": f1_score(y_val, y_pred),
        "auc": roc_auc_score(y_val, y_proba),
        "model": model,
    }


def evaluate_model_cv(
    model: ClassifierMixin,
    x: pd.DataFrame,
    y: pd.Series,
    cv: int = 5,
) -> dict:
    """Evaluates a model using cross-validation and returns performance metrics."""
    y_pred = cross_val_predict(model, x, y, cv=cv)
    y_proba = cross_val_predict(model, x, y, cv=cv, method="predict_proba")[:, 1]
    auc = roc_auc_score(y, y_proba)

    return {
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred),
        "recall": recall_score(y, y_pred),
        "f1": f1_score(y, y_pred),
        "auc": auc,
    }


def evaluate_pipeline(
    pipeline: Pipeline,
    x_train: pd.DataFrame,
    x_val: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
) -> dict:
    """Evaluates a pipeline's performance on validation data and returns metrics."""
    pipeline.fit(x_train, y_train)

    y_pred = pipeline.predict(x_val)
    y_proba = pipeline.predict_proba(x_val)[:, 1]

    return {
        "accuracy": accuracy_score(y_val, y_pred),
        "precision": precision_score(y_val, y_pred),
        "recall": recall_score(y_val, y_pred),
        "f1": f1_score(y_val, y_pred),
        "auc": roc_auc_score(y_val, y_proba),
        "pipeline": pipeline,
    }


def create_model_comparison_table(results: dict) -> pd.DataFrame:
    """Creates a styled DataFrame comparing model performance metrics."""
    comparison_df = pd.DataFrame(
        {
            "Model": list(results.keys()),
            "Accuracy": [results[m]["accuracy"] for m in results],
            "Precision": [results[m]["precision"] for m in results],
            "Recall": [results[m]["recall"] for m in results],
            "F1 Score": [results[m]["f1"] for m in results],
            "AUC": [results[m]["auc"] for m in results],
        }
    ).sort_values("F1 Score", ascending=False)

    return comparison_df.style.background_gradient(
        subset=["Accuracy", "Precision", "Recall", "F1 Score", "AUC"]
    )


def print_single_result(result: dict) -> pd.DataFrame:
    """Prints a single result in a DataFrame."""
    return pd.DataFrame(
        {
            "Model": ["XGBoost"],
            "Accuracy": [result["accuracy"]],
            "Precision": [result["precision"]],
            "Recall": [result["recall"]],
            "F1 Score": [result["f1"]],
            "AUC": [result["auc"]],
        }
    )
