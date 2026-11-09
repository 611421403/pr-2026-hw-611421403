import numpy as np


class GaussianNBScratch:
    """Gaussian Naive Bayes implemented with NumPy only."""

    def __init__(self, var_smoothing: float = 1e-9):
        if var_smoothing < 0:
            raise ValueError("var_smoothing must be non-negative.")
        self.var_smoothing = float(var_smoothing)
        self.classes_ = None
        self.class_prior_ = None
        self.theta_ = None
        self.var_ = None
        self.epsilon_ = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)

        if X.ndim != 2:
            raise ValueError("X must be a 2-D array.")
        if y.ndim != 1 or len(y) != len(X):
            raise ValueError("y must be 1-D and have the same number of rows as X.")
        if len(X) == 0:
            raise ValueError("Training data cannot be empty.")
        if not np.all(np.isfinite(X)):
            raise ValueError("X contains NaN or infinite values.")

        self.classes_, counts = np.unique(y, return_counts=True)
        n_classes = len(self.classes_)
        n_features = X.shape[1]

        self.class_prior_ = counts.astype(np.float64) / len(y)
        self.theta_ = np.zeros((n_classes, n_features), dtype=np.float64)
        self.var_ = np.zeros((n_classes, n_features), dtype=np.float64)

        for idx, cls in enumerate(self.classes_):
            X_cls = X[y == cls]
            self.theta_[idx] = np.mean(X_cls, axis=0)
            # Maximum-likelihood variance: denominator is n, not n-1.
            self.var_[idx] = np.var(X_cls, axis=0, ddof=0)

        # Match sklearn GaussianNB's global variance-scaled smoothing idea.
        global_max_var = float(np.max(np.var(X, axis=0, ddof=0)))
        self.epsilon_ = self.var_smoothing * global_max_var
        self.var_ += self.epsilon_
        self.var_ = np.maximum(self.var_, np.finfo(np.float64).tiny)

        return self

    def predict_log_proba(self, X):
        if self.classes_ is None:
            raise RuntimeError("Call fit before prediction.")

        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != self.theta_.shape[1]:
            raise ValueError("X has an incompatible shape.")
        if not np.all(np.isfinite(X)):
            raise ValueError("X contains NaN or infinite values.")

        # log P(x|class) = sum_j log N(x_j | mean_j, variance_j).
        # Work in log space to avoid multiplying many tiny probabilities.
        log_likelihood = -0.5 * np.sum(
            np.log(2.0 * np.pi * self.var_)[None, :, :]
            + ((X[:, None, :] - self.theta_[None, :, :]) ** 2)
            / self.var_[None, :, :],
            axis=2,
        )
        joint_log = log_likelihood + np.log(self.class_prior_)[None, :]

        # Stable log-sum-exp normalization.
        row_max = np.max(joint_log, axis=1, keepdims=True)
        log_normalizer = row_max + np.log(
            np.sum(np.exp(joint_log - row_max), axis=1, keepdims=True)
        )
        return joint_log - log_normalizer

    def predict(self, X):
        log_proba = self.predict_log_proba(X)
        return self.classes_[np.argmax(log_proba, axis=1)]
