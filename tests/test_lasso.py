import numpy as np
from lasso.lasso import LassoHomotopy

def test_basic_fit():
    np.random.seed(42)
    X = np.random.randn(100, 5)
    true_coef = np.array([2, 0, 0, 3, 0])
    y = X @ true_coef + np.random.randn(100) * 0.1

    model = LassoHomotopy(alpha=0.1)
    model.fit(X, y)
    assert len(model.coef_) == 5
    assert np.abs(model.coef_[0]) > 0.5
    assert np.abs(model.coef_[3]) > 0.5

def test_collinear_data():
    np.random.seed(0)
    X = np.random.randn(100, 2)
    X = np.hstack([X, X[:, [0]] * 1.0])  # Make X[:,2] collinear with X[:,0]
    y = 3 * X[:, 0] + np.random.randn(100) * 0.1

    model = LassoHomotopy(alpha=0.5)
    model.fit(X, y)
    # Should pick one of collinear features, zero the other
    non_zero_count = np.sum(np.abs(model.coef_) > 1e-2)
    assert non_zero_count <= 2  # Only one or two non-zero due to collinearity
