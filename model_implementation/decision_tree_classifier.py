"""
Decision Tree Classifier implementation for sleep debt category classification.

This module wraps scikit-learn's DecisionTreeClassifier inside a clean class
that uses the shared data service for loading and preprocessing the dataset.
"""

from __future__ import annotations

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from model_implementation.service.data_service import get_preprocessed_data


class DecisionTreeModel:
    """
    Decision Tree classifier for sleep debt category prediction.

    The model predicts one of four sleep debt categories:
        - Mild Deficit
        - Moderate Debt
        - Optimal Recovery
        - Severe Sleep Debt

    Decision Trees are non-parametric models that partition the feature space
    using axis-aligned splits. They do not require feature scaling, so
    scale_features defaults to False.

    Attributes:
        model (DecisionTreeClassifier): The underlying scikit-learn estimator.
        data (dict | None): Preprocessed data dict returned by the data service.
                            Populated after calling load_data().
        is_trained (bool): True once fit() has been called successfully.

    Example usage:
        >>> dt = DecisionTreeModel()
        >>> dt.load_data()
        >>> dt.fit()
        >>> metrics = dt.evaluate()
        >>> print(metrics["accuracy"])
    """

    def __init__(
        self,
        criterion: str = "gini",
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: str | int | float | None = None,
        random_state: int = 42,
        test_size: float = 0.2,
        scale_features: bool = False,
    ) -> None:
        """
        Initialise the Decision Tree model.

        Args:
            criterion:          Splitting quality measure. Options: 'gini', 'entropy',
                                'log_loss' (default 'gini').
            max_depth:          Maximum depth of the tree. None means nodes are expanded
                                until all leaves are pure or contain fewer than
                                min_samples_split samples (default None).
            min_samples_split:  Minimum number of samples required to split an internal
                                node (default 2).
            min_samples_leaf:   Minimum number of samples required to be at a leaf node
                                (default 1).
            max_features:       Number of features considered when looking for the best
                                split. Options: int, float, 'sqrt', 'log2', or None
                                (default None — use all features).
            random_state:       Seed for reproducibility (default 42).
            test_size:          Fraction of the dataset used for testing (default 0.2).
            scale_features:     Whether to standardise features before training.
                                Defaults to False — Decision Trees are scale-invariant.
        """
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.test_size = test_size
        self.scale_features = scale_features

        # Build the scikit-learn estimator (not yet trained).
        self.model = DecisionTreeClassifier(
            criterion=self.criterion,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
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

    def fit(self) -> DecisionTreeModel:
        """
        Train the decision tree on the training split.

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

        Probabilities are computed as the fraction of training samples of each
        class that reach the same leaf as the input sample.

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

    def get_feature_importances(self) -> dict:
        """
        Return feature importances (Gini/entropy-based) mapped to feature names.

        Importances are normalised so they sum to 1. A higher value means the
        feature contributes more to the splitting decisions across the tree.

        Raises:
            RuntimeError: If the model has not been trained yet.

        Returns:
            A dict mapping feature name -> importance score, sorted descending.
        """
        if not self.is_trained:
            raise RuntimeError("Call fit() before get_feature_importances().")

        feature_names = self.data["feature_names"]
        importances = self.model.feature_importances_

        return dict(
            sorted(
                zip(feature_names, importances),
                key=lambda item: item[1],
                reverse=True,
            )
        )

    def __repr__(self) -> str:
        status = "trained" if self.is_trained else "untrained"
        return (
            f"DecisionTreeModel("
            f"criterion='{self.criterion}', max_depth={self.max_depth}, "
            f"status={status})"
        )
