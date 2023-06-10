from sklearn import datasets
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
import matplotlib.pyplot as plt
#
# Load the california Data Set
#
bh = fetch_california_housing()
X = bh.data
y = bh.target
#
# Create training and test split
#
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
#
# Create an instance of Lasso Regression implementation
# Regularization parameter 1.0
lasso = Lasso(alpha=1.0)
#
# Fit the Lasso model
#
lasso.fit(X_train, y_train)
#
# Create the model score
#
score_test = lasso.score(X_test, y_test)
score_train = lasso.score(X_train, y_train)

# Make predictions on the test set
y_pred = lasso.predict(X_test)

# Plot true values vs predicted values
plt.scatter(y_test, y_pred)
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('True Values vs Predicted Values')
plt.show()

print("Test score:", score_test)
print("Train score:", score_train)
