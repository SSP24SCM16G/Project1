TEAM MEMBERS:
1. SAMHITHA GOURU A20550016
2. HARINI KORADA A20546869
3. MAITHREYEE DOMA A20549555
4. M. GAYTHRI RISHITHA A20563808
# LASSO Homotopy Regression

## What does the model you have implemented do and when should it be used?
This model implements LASSO (Least Absolute Shrinkage and Selection Operator) regression using the Homotopy Method. 
It performs linear regression with L1 regularization, encouraging sparsity in the solution. 
It should be used when feature selection is important or when handling highly collinear data.

## How did you test your model to determine if it is working reasonably correctly?
- I created synthetic datasets with known coefficients.
- I tested with collinear data to check if the model produces sparse solutions.
- I wrote PyTest unit tests to ensure correct behavior.

## What parameters have you exposed to users of your implementation in order to tune performance?
- `alpha`: Regularization strength parameter to control sparsity.

## Are there specific inputs that your implementation has trouble with? Given more time, could you work around these or is it fundamental?
- Large datasets may slow down performance due to the implementation's complexity.
- Non-linear relationships cannot be captured since the model is linear.
- Given more time, performance improvements and automatic tuning of alpha could be added.

## How to run the code
1. Create and activate a virtual environment:
```
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

2. Install requirements:
```
pip install -r requirements.txt
```

3. Run tests:
```
pytest tests/
```

## Example Usage
```
from lasso.lasso import LassoHomotopy
import numpy as np

X = np.random.randn(100, 10)
y = X @ np.array([1.5, -2, 0, 0, 3, 0, 0, 0, 0, 0]) + np.random.randn(100) * 0.1
lasso = LassoHomotopy(alpha=0.5)
lasso.fit(X, y)
print("Coefficients:", lasso.coef_)
```
