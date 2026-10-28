"""
Logistic Regression model implementation for sleep debt category classification.

This module wraps scikit-learn's LogisticRegression inside a clean class
that uses the shared data service for loading and preprocessing the dataset.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from model_implementation.service.data_service import get_preprocessed_data


class LogisticRegressionModel:
    """
    Logistic Regression classifier for sleep debt category prediction.

    The model predicts one of four sleep debt categories:
        - Mild Deficit
        - Moderate Debt
        - Optimal Recovery
        - Severe Sleep Debt

    scikit-learn's LogisticRegression handles multi-class problems automatically
    (using a one-vs-rest or multinomial strategy depending on the solver), so no
    explicit multi_class argument is needed.

    Attributes:
        model (LogisticRegression): The underlying scikit-learn estimator.
        data (dict | None): Preprocessed data dict returned by the data service.
                            Populated after calling load_data().
        is_trained (bool): True once fit() has been called successfully.

    Example usage:
        >>> lr = LogisticRegressionModel()
        >>> lr.load_data()
        >>> lr.fit()
        >>> metrics = lr.evaluate()
        >>> print(metrics["accuracy"])
    """

    def __init__(
        self,
        C: float = 1.0,
        max_iter: int = 1000,
        solver: str = "lbfgs",
        random_state: int = 42,
        test_size: float = 0.2,
        scale_features: bool = True,
    ) -> None:
        """
        Initialise the Logistic Regression model.

        Args:
            C:             Inverse of regularisation strength; smaller values
                           apply stronger regularisation (default 1.0).
            max_iter:      Maximum number of iterations for the solver (default 1000).
            solver:        Algorithm for the optimisation problem.
                           Options: 'lbfgs', 'saga', 'newton-cg', 'sag', 'liblinear'
                           (default 'lbfgs').
                           Note: 'lbfgs' natively supports multi-class classification.
            random_state:  Seed for reproducibility (default 42).
            test_size:     Fraction of the dataset used for testing (default 0.2).
            scale_features: Whether to standardise features before training
                            (default True, recommended for logistic regression).
        """
        self.C = C
        self.max_iter = max_iter
        self.solver = solver
        self.random_state = random_state
        self.test_size = test_size
        self.scale_features = scale_features

        # Build the scikit-learn estimator (not yet trained).
        # multi_class was deprecated in sklearn 1.5 and removed in 1.6 —
        # the solver now handles multi-class automatically.
        self.model = LogisticRegression(
            C=self.C,
            max_iter=self.max_iter,
            solver=self.solver,
            random_state=self.random_state,
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

    def fit(self) -> LogisticRegressionModel:
        """
        Train the logistic regression model on the training split.

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
        probabilities, which requires the model to support predict_proba().

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

    def get_coefficients(self) -> dict:
        """
        Return the model coefficients mapped to their feature names.

        Raises:
            RuntimeError: If the model has not been trained yet.

        Returns:
            A dictionary with:
                - "classes"      : list of class names (decoded).
                - "coefficients" : dict mapping class_name -> {feature: coef}.
        """
        if not self.is_trained:
            raise RuntimeError("Call fit() before get_coefficients().")

        le = self.data["label_encoder"]
        feature_names = self.data["feature_names"]
        class_names = le.inverse_transform(self.model.classes_).tolist()

        coefficients = {
            class_name: dict(zip(feature_names, coefs))
            for class_name, coefs in zip(class_names, self.model.coef_)
        }

        return {
            "classes": class_names,
            "coefficients": coefficients,
        }

    def __repr__(self) -> str:
        status = "trained" if self.is_trained else "untrained"
        return (
            f"LogisticRegressionModel("
            f"C={self.C}, solver='{self.solver}', "
            f"max_iter={self.max_iter}, status={status})"
        )
