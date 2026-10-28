"""
Gaussian Naive Bayes Classifier implementation for sleep debt category classification.

This module wraps scikit-learn's GaussianNB inside a clean class that uses the
shared data service for loading and preprocessing the dataset.
"""

from __future__ import annotations

import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from model_implementation.service.data_service import get_preprocessed_data


class GaussianNaiveBayesModel:
    """
    Gaussian Naive Bayes classifier for sleep debt category prediction.

    The model predicts one of four sleep debt categories:
        - Mild Deficit
        - Moderate Debt
        - Optimal Recovery
        - Severe Sleep Debt

    Gaussian Naive Bayes assumes each feature follows a Gaussian (normal)
    distribution within each class and that features are conditionally
    independent given the class. It is a fast, probabilistic classifier that
    works well even with relatively small datasets.

    Attributes:
        model (GaussianNB): The underlying scikit-learn estimator.
        data (dict | None): Preprocessed data dict returned by the data service.
                            Populated after calling load_data().
        is_trained (bool): True once fit() has been called successfully.

    Example usage:
        >>> gnb = GaussianNaiveBayesModel()
        >>> gnb.load_data()
        >>> gnb.fit()
        >>> metrics = gnb.evaluate()
        >>> print(metrics["accuracy"])
    """

    def __init__(
        self,
        var_smoothing: float = 1e-9,
        test_size: float = 0.2,
        scale_features: bool = True,
    ) -> None:
        """
        Initialise the Gaussian Naive Bayes model.

        Args:
            var_smoothing:  Portion of the largest variance of all features that
                            is added to variances for numerical stability. A larger
                            value increases the smoothing effect and acts as a
                            regulariser (default 1e-9).
            test_size:      Fraction of the dataset used for testing (default 0.2).
            scale_features: Whether to standardise features before training.
                            Defaults to True — scaling helps GaussianNB fit more
                            accurate per-class Gaussian distributions.
        """
        self.var_smoothing = var_smoothing
        self.test_size = test_size
        self.scale_features = scale_features

        # Build the scikit-learn estimator (not yet trained).
        self.model = GaussianNB(
            var_smoothing=self.var_smoothing,
        )

        self.data: dict | None = None
        self.is_trained: bool = False

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------

    def load_data(self) -> None:
        """
        Load and preprocess the dataset using the shared data service.

        After this call, self.data will contain:
            - X_train, X_test  : Feature matrices (numpy arrays)
            - y_train, y_test  : Encoded label arrays
            - feature_names    : List of feature column names
            - label_encoder    : Fitted LabelEncoder for decoding predictions
            - scaler           : Fitted StandardScaler (or None)
        """
        self.data = get_preprocessed_data(
            test_size=self.test_size,
            scale_features=self.scale_features,
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self) -> GaussianNaiveBayesModel:
        """
        Train the Gaussian Naive Bayes model on the training split.

        Raises:
            RuntimeError: If load_data() has not been called first.

        Returns:
            self  (to allow method chaining)
        """
        if self.data is None:
            raise RuntimeError("Call load_data() before fit().")

        self.model.fit(self.data["X_train"], self.data["y_train"])
        self.is_trained = True
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for the given feature matrix.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
               Should be preprocessed (scaled) the same way as training data.

        Raises:
            RuntimeError: If the model has not been trained yet.

        Returns:
            Encoded label array of shape (n_samples,).
            Use self.data["label_encoder"].inverse_transform() to get
            human-readable category names.
        """
        if not self.is_trained:
            raise RuntimeError("Call fit() before predict().")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return class probability estimates for the given feature matrix.

        Probabilities are computed via Bayes' theorem using per-class Gaussian
        likelihoods and the prior class probabilities learned from training data.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Raises:
            RuntimeError: If the model has not been trained yet.

        Returns:
            Probability array of shape (n_samples, n_classes).
            Column order matches self.model.classes_.
        """
        if not self.is_trained:
            raise RuntimeError("Call fit() before predict_proba().")
        return self.model.predict_proba(X)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self) -> dict:
        """
        Evaluate the trained model on the held-out test set.

        All multi-class metrics use weighted averaging (weighted by class support),
        so they account for class imbalance.

        AUC is computed with the One-vs-Rest (OvR) strategy using predicted
        probabilities.

        Raises:
            RuntimeError: If the model has not been trained yet.

        Returns:
            A dictionary with the following keys:
                - "accuracy"  : float — fraction of correctly classified samples.
                - "auc_score" : float — weighted-average AUC (One-vs-Rest, multi-class).
                - "precision" : float — weighted-average precision across all classes.
                - "recall"    : float — weighted-average recall across all classes.
                - "f1_score"  : float — weighted-average F1 score across all classes.
                - "mcc_score" : float — Matthews Correlation Coefficient (MCC).
                                        Ranges from -1 (worst) to +1 (perfect).
        """
        if not self.is_trained:
            raise RuntimeError("Call fit() before evaluate().")

        X_test = self.data["X_test"]
        y_test = self.data["y_test"]

        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)

        return {
            "accuracy": accuracy_score(y_test, y_pred),
            "auc_score": roc_auc_score(y_test, y_proba, multi_class="ovr", average="weighted"),
            "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
            "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
            "f1_score": f1_score(y_test, y_pred, average="weighted", zero_division=0),
            "mcc_score": matthews_corrcoef(y_test, y_pred),
        }

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------

    def get_class_statistics(self) -> dict:
        """
        Return the per-class Gaussian parameters learned during training.

        GaussianNB fits a Gaussian distribution (mean and variance) for each
        feature in each class. These statistics drive the likelihood computation.

        Raises:
            RuntimeError: If the model has not been trained yet.

        Returns:
            A dictionary with:
                - "classes"   : list of decoded class name strings.
                - "priors"    : dict mapping class_name -> prior probability.
                - "means"     : dict mapping class_name -> {feature: mean}.
                - "variances" : dict mapping class_name -> {feature: variance}.
        """
        if not self.is_trained:
            raise RuntimeError("Call fit() before get_class_statistics().")

        le = self.data["label_encoder"]
        feature_names = self.data["feature_names"]
        class_names = le.inverse_transform(self.model.classes_).tolist()

        priors = dict(zip(class_names, self.model.class_prior_.tolist()))

        means = {
            class_name: dict(zip(feature_names, mu))
            for class_name, mu in zip(class_names, self.model.theta_)
        }

        variances = {
            class_name: dict(zip(feature_names, var))
            for class_name, var in zip(class_names, self.model.var_)
        }

        return {
            "classes": class_names,
            "priors": priors,
            "means": means,
            "variances": variances,
        }

    def __repr__(self) -> str:
        status = "trained" if self.is_trained else "untrained"
        return (
            f"GaussianNaiveBayesModel("
            f"var_smoothing={self.var_smoothing}, "
            f"status={status})"
        )
