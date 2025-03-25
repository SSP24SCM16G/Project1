import numpy as np

class LassoHomotopy:
    def __init__(self, alpha=1.0, tol=1e-4, max_iter=1000):
        self.alpha = alpha
        self.tol = tol
        self.max_iter = max_iter
        self.coef_ = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        X = np.array(X)
        y = np.array(y)

        # Initialize
        residual = y.copy()
        self.coef_ = np.zeros(n_features)
        active_set = []
        sign_active = []
        iteration = 0

        while iteration < self.max_iter:
            # Compute correlations
            corr = X.T @ residual
            c_max = np.max(np.abs(corr))

            # Check stopping condition
            if c_max <= self.alpha + self.tol:
                break

            # Add variable to active set
            j_max = np.argmax(np.abs(corr))
            if j_max not in active_set:
                active_set.append(j_max)
                sign_active.append(np.sign(corr[j_max]))

            # Build active matrix
            X_active = X[:, active_set]
            s = np.array(sign_active)

            # Solve least squares problem for active set
            try:
                w = np.linalg.pinv(X_active.T @ X_active) @ (X_active.T @ y - self.alpha * s / 2)
            except np.linalg.LinAlgError:
                break 

            # Update coefficients
            self.coef_[active_set] = w

            # Update residual
            residual = y - X_active @ w

            iteration += 1

        return self

    def predict(self, X):
        return X @ self.coef_
