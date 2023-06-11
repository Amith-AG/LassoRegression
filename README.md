# LassoRegression

The code provided demonstrates the usage of Lasso regression for the California housing dataset. Lasso regression is a linear regression technique that performs both feature selection and regularization by adding a penalty term to the ordinary least squares objective function. It encourages sparsity in the model by shrinking some coefficients to exactly zero.

Here is a step-by-step explanation of the code:

1. Import necessary libraries: The code starts by importing the required libraries, including `datasets` and `Lasso` from `sklearn` for the dataset and Lasso regression implementation, respectively. It also imports `train_test_split` for splitting the data into training and testing sets, and `matplotlib.pyplot` for plotting.

2. Load the California housing dataset: The code uses the `fetch_california_housing` function from `datasets` to load the California housing dataset. The input features are stored in `X`, and the target variable (housing prices) is stored in `y`.

3. Create a training and test split: The `train_test_split` function is used to split the data into training and testing sets. It takes the input features (`X`) and target variable (`y`) along with the desired test size (30% in this case) and a random seed (42) for reproducibility. The resulting split data is stored in `X_train`, `X_test`, `y_train`, and `y_test`.

4. Create an instance of Lasso regression: The code creates an instance of the Lasso regression model using `Lasso(alpha=1.0)`. The `alpha` parameter controls the strength of the regularization. Higher values of `alpha` result in more regularization and can shrink more coefficients to zero.

5. Fit the Lasso model: The `fit` method is used to train the Lasso regression model. It takes the training data (`X_train` and `y_train`) as input and adjusts the model's coefficients to fit the training data.

6. Calculate the model score: The code calculates the coefficient of determination (R^2 score) for both the test and training data using the `score` method. It compares the predicted values from the Lasso model with the actual target values. The resulting scores are stored in `score_test` and `score_train`, respectively.

7. Make predictions on the test set: The `predict` method is used to make predictions on the test set (`X_test`) using the trained Lasso model. The predicted values are stored in `y_pred`.

8. Plot true values vs predicted values: The code uses `plt.scatter` to create a scatter plot comparing the true values (`y_test`) with the predicted values (`y_pred`). This plot visualizes how well the model's predictions align with the actual values.

9. Display the scores: The code prints the test and train scores using `print("Test score:", score_test)` and `print("Train score:", score_train)`. These scores indicate the model's performance on the test and training data, respectively.

Overall, this code demonstrates how to train a Lasso regression model on the California housing dataset, evaluate its performance, make predictions, and visualize the results.
