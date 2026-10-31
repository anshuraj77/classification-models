"""
Support Vector Machine Classifier implementation for sleep debt category classification.

This module wraps scikit-learn's SVC inside a clean class that uses the shared
data service for loading and preprocessing the dataset.
"""

from __future__ import annotations

import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from model_implementation.service.data_service import get_preprocessed_data


class SupportVectorMachineModel:
    """
    Support Vector Machine classifier for sleep debt category prediction.

    The model predicts one of four sleep debt categories:
        - Mild Deficit
        - Moderate Debt
        - Optimal Recovery
        - Severe Sleep Debt

    SVM finds the optimal hyperplane that maximises the margin between classes.
    For multi-class problems, scikit-learn uses a One-vs-One (OvO) strategy by
    default — one binary classifier is trained for every pair of classes and the
    class that wins the most duels is chosen.

    probability=True is enabled so that predict_proba() and AUC scoring are
    available. This uses Platt scaling (an internal cross-validation during
    fit()), which adds a small overhead.

    Because SVM is distance-based, feature scaling is essential — scale_features
    defaults to True.

    Attributes:
        model (SVC): The underlying scikit-learn estimator.
        data (dict | None): Preprocessed data dict returned by the data service.
                            Populated after calling load_data().
        is_trained (bool): True once fit() has been called successfully.

    Example usage:
        >>> svm = SupportVectorMachineModel()
        >>> svm.load_data()
        >>> svm.fit()
        >>> metrics = svm.evaluate()
        >>> print(metrics["accuracy"])
    """

    def __init__(
        self,
        C: float = 1.0,
        kernel: str = "rbf",
        gamma: str | float = "scale",
        degree: int = 3,
        coef0: float = 0.0,
        random_state: int = 42,
        test_size: float = 0.2,
        scale_features: bool = True,
    ) -> None:
        """
        Initialise the Support Vector Machine model.

        Args:
            C:              Regularisation parameter. Smaller values allow a
                            wider margin at the cost of more misclassifications
                            (softer margin); larger values enforce stricter
                            classification (harder margin). (default 1.0).
            kernel:         Kernel function used to map inputs into a higher-
                            dimensional feature space. Options:
                            - 'rbf'    : Radial Basis Function (default) — works
                                         well for most non-linear problems.
                            - 'linear' : Suitable when data is linearly separable.
                            - 'poly'   : Polynomial kernel; degree controls order.
                            - 'sigmoid': Hyperbolic tangent kernel.
            gamma:          Kernel coefficient for 'rbf', 'poly', and 'sigmoid'.
                            - 'scale'  : 1 / (n_features * X.var()) (default).
                            - 'auto'   : 1 / n_features.
                            - float    : explicit value.
            degree:         Degree of the polynomial kernel ('poly' only).
                            Ignored by other kernels (default 3).
            coef0:          Independent term in 'poly' and 'sigmoid' kernels
                            (default 0.0).
            random_state:   Seed for reproducibility in probability estimation
                            (Platt scaling) (default 42).
            test_size:      Fraction of the dataset used for testing (default 0.2).
            scale_features: Whether to standardise features before training.
                            Defaults to True — SVM is sensitive to feature
                            magnitude; unscaled data will degrade performance.
        """
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.random_state = random_state
        self.test_size = test_size
        self.scale_features = scale_features

        # Build the scikit-learn estimator (not yet trained).
        # probability=True enables predict_proba() via Platt scaling.
        self.model = SVC(
            C=self.C,
            kernel=self.kernel,
            gamma=self.gamma,
            degree=self.degree,
            coef0=self.coef0,
            probability=True,
            random_state=self.random_state,
        )

        self.data: dict | None = None
        self.is_trained: bool = False

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------

    def load_data(self, file_path: str | None = None) -> None:
        """
        Load and preprocess the dataset using the shared data service.
        Optionally, pass a CSV path instead of using the default dataset.

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
            **({"file_path": file_path} if file_path is not None else {}),
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self) -> SupportVectorMachineModel:
        """
        Train the SVM on the training split.

        Because probability=True is set, Platt scaling is applied internally
        during fit() via cross-validation, which may increase training time
        compared to a plain SVC.

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

        Probabilities are computed using Platt scaling fitted during training.
        Note: these are calibrated probabilities, not raw SVM decision scores.

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

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Return the raw signed distance of samples to the decision hyperplane.

        For multi-class (OvO), returns one score per class pair. These are
        the raw SVM scores before Platt scaling.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Raises:
            RuntimeError: If the model has not been trained yet.

        Returns:
            Decision scores of shape (n_samples, n_classes * (n_classes - 1) / 2)
            for OvO multi-class, or (n_samples,) for binary classification.
        """
        if not self.is_trained:
            raise RuntimeError("Call fit() before decision_function().")
        return self.model.decision_function(X)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self) -> dict:
        """
        Evaluate the trained model on the held-out test set.

        All multi-class metrics use weighted averaging (weighted by class support),
        so they account for class imbalance.

        AUC is computed with the One-vs-Rest (OvR) strategy using Platt-scaled
        probabilities from predict_proba().

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

    def get_support_vectors(self) -> dict:
        """
        Return the support vectors and their per-class counts.

        Support vectors are the training samples that lie on or within the
        margin boundary and directly influence the position of the hyperplane.

        Only available for kernel='linear', 'rbf', 'poly', or 'sigmoid'.

        Raises:
            RuntimeError: If the model has not been trained yet.

        Returns:
            A dictionary with:
                - "support_vectors"       : np.ndarray of shape
                                            (n_support_vectors, n_features).
                - "n_support_per_class"   : dict mapping class_name -> count of
                                            support vectors for that class.
                - "support_vector_indices": np.ndarray — indices of support
                                            vectors in the original training set.
        """
        if not self.is_trained:
            raise RuntimeError("Call fit() before get_support_vectors().")

        le = self.data["label_encoder"]
        class_names = le.inverse_transform(self.model.classes_).tolist()

        n_support_per_class = dict(zip(class_names, self.model.n_support_.tolist()))

        return {
            "support_vectors": self.model.support_vectors_,
            "n_support_per_class": n_support_per_class,
            "support_vector_indices": self.model.support_,
        }

    def __repr__(self) -> str:
        status = "trained" if self.is_trained else "untrained"
        return (
            f"SupportVectorMachineModel("
            f"kernel='{self.kernel}', C={self.C}, "
            f"gamma='{self.gamma}', status={status})"
        )
