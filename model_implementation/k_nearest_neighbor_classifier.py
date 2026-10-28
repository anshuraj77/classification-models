"""
K-Nearest Neighbors Classifier implementation for sleep debt category classification.

This module wraps scikit-learn's KNeighborsClassifier inside a clean class
that uses the shared data service for loading and preprocessing the dataset.
"""

from __future__ import annotations

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from model_implementation.service.data_service import get_preprocessed_data


class KNearestNeighborModel:
    """
    K-Nearest Neighbors classifier for sleep debt category prediction.

    The model predicts one of four sleep debt categories:
        - Mild Deficit
        - Moderate Debt
        - Optimal Recovery
        - Severe Sleep Debt

    KNN is a non-parametric, instance-based algorithm that classifies a sample
    by majority vote among its k nearest neighbors in the feature space.
    Because it relies on distance calculations, feature scaling is essential
    for meaningful neighbor comparisons — scale_features defaults to True.

    Attributes:
        model (KNeighborsClassifier): The underlying scikit-learn estimator.
        data (dict | None): Preprocessed data dict returned by the data service.
                            Populated after calling load_data().
        is_trained (bool): True once fit() has been called successfully.

    Example usage:
        >>> knn = KNearestNeighborModel()
        >>> knn.load_data()
        >>> knn.fit()
        >>> metrics = knn.evaluate()
        >>> print(metrics["accuracy"])
    """

    def __init__(
        self,
        n_neighbors: int = 5,
        weights: str = "uniform",
        metric: str = "minkowski",
        p: int = 2,
        algorithm: str = "auto",
        test_size: float = 0.2,
        scale_features: bool = True,
    ) -> None:
        """
        Initialise the K-Nearest Neighbors model.

        Args:
            n_neighbors:    Number of neighbors to use for classification
                            (default 5).
            weights:        Weight function used in prediction.
                            - 'uniform'  : all neighbors vote equally.
                            - 'distance' : closer neighbors have higher influence.
                            (default 'uniform').
            metric:         Distance metric for the tree. Common choices:
                            'minkowski' (generalises Euclidean/Manhattan),
                            'euclidean', 'manhattan', 'cosine' (default 'minkowski').
            p:              Power parameter for the Minkowski metric.
                            p=1 → Manhattan distance, p=2 → Euclidean distance
                            (default 2).
            algorithm:      Algorithm used to compute nearest neighbors.
                            Options: 'auto', 'ball_tree', 'kd_tree', 'brute'
                            (default 'auto' — lets sklearn choose based on data).
            test_size:      Fraction of the dataset used for testing (default 0.2).
            scale_features: Whether to standardise features before training.
                            Defaults to True — KNN is distance-based so scaling
                            is essential to prevent high-magnitude features from
                            dominating the distance calculation.
        """
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.metric = metric
        self.p = p
        self.algorithm = algorithm
        self.test_size = test_size
        self.scale_features = scale_features

        # Build the scikit-learn estimator (not yet trained).
        self.model = KNeighborsClassifier(
            n_neighbors=self.n_neighbors,
            weights=self.weights,
            metric=self.metric,
            p=self.p,
            algorithm=self.algorithm,
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

    def fit(self) -> KNearestNeighborModel:
        """
        Fit the KNN model on the training split.

        KNN is a lazy learner — no explicit model is built during fit().
        The training data is simply stored and used at prediction time to
        find the k nearest neighbors.

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
               Must be scaled the same way as the training data.

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

        Probabilities are computed as the fraction of the k neighbors that
        belong to each class (optionally weighted by inverse distance).

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

    def get_neighbors(self, X: np.ndarray) -> dict:
        """
        Return the k nearest neighbors for each sample in X.

        Useful for inspecting which training samples most influence a prediction.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
               Must be scaled the same way as the training data.

        Raises:
            RuntimeError: If the model has not been trained yet.

        Returns:
            A dictionary with:
                - "distances" : np.ndarray of shape (n_samples, n_neighbors) —
                                distance from each query point to each neighbor.
                - "indices"   : np.ndarray of shape (n_samples, n_neighbors) —
                                indices into the training set for each neighbor.
        """
        if not self.is_trained:
            raise RuntimeError("Call fit() before get_neighbors().")

        distances, indices = self.model.kneighbors(X)
        return {
            "distances": distances,
            "indices": indices,
        }

    def __repr__(self) -> str:
        status = "trained" if self.is_trained else "untrained"
        return (
            f"KNearestNeighborModel("
            f"n_neighbors={self.n_neighbors}, weights='{self.weights}', "
            f"metric='{self.metric}', status={status})"
        )
