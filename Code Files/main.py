# importing pandas
import pandas as pd

# importing Required Machine learning Models (Linear Regression, Decision Trees, Random Forest, etc)
from sklearn.model_selection import train_test_split
# Linear Regression
from sklearn.linear_model import LinearRegression
# Decision Trees 
from sklearn.tree import DecisionTreeRegressor
# Random Forest Regressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
# metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error

# Accuracy Calculating
def calculate_accuracy(y_true, y_pred):
    # mean absolute error
    mae = mean_absolute_error(y_true, y_pred)
    # Accuracy
    accuracy = ((y_true.mean() - mae) / y_true.mean()) * 100
    # accuracy return
    return accuracy

# Load the dataset
# This DataSet Contains Data of 2000+ employees it will be used for training and testing purposes
data = pd.read_csv('Salary Prediction of Data Professions.csv')

# Handle missing values for numeric columns only
numeric_cols = data.select_dtypes(include=['number']).columns
# numeric cols
data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())

# One-hot encode categorical columns
data = pd.get_dummies(data, columns=['SEX', 'DESIGNATION', 'UNIT'])

# Check if 'SALARY' column is present in the DataFrame
if 'SALARY' not in data.columns:
    # error message 
    print("Error: 'SALARY' column not found in the dataset.")
    
    exit(1)  # Exit the program with an error code

# Split the data into features and target
X = data.drop(columns=['SALARY', 'FIRST NAME', 'LAST NAME', 'DOJ', 'CURRENT DATE'])
# Y Data
y = data['SALARY']

# Split the data into training and testing sets
# Some Data will be used for training the ML models and some will be used for testing the model performance based on the accuracy and efficieny of the model the best suited will be chosen for showing the reuslt accuracy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and evaluate the models
models = {
    # Linear Regression 
    'Linear Regression': LinearRegression(),
    # Decision Trees
    'Decision Tree': DecisionTreeRegressor(),
    # Random Forest
    'Random Forest': RandomForestRegressor(),
    # Gradient Boosting
    'Gradient Boosting': GradientBoostingRegressor()
}
# best model initialized to None
best_model = None
# rmse inf
best_rmse = float('inf')  # Initialize with a high value

# Models Training 
for name, model in models.items():
    # X and Y Model 
    model.fit(X_train, y_train)
    # Model Predict
    y_pred = model.predict(X_test)
    # root_mean
    rmse = root_mean_squared_error(y_test, y_pred)
    
    if rmse < best_rmse:
        # best model initialization
        best_model = model
        # rmse 
        best_rmse = rmse

# Print the evaluation metrics for the best-performing model
# All Models have performed there analysis all of them will be evaluated and then best will be chosen
y_pred_best = best_model.predict(X_test)
# mae best
mae_best = mean_absolute_error(y_test, y_pred_best)
# mse best
mse_best = mean_squared_error(y_test, y_pred_best)
# Score and Calculate Accuracy
rmse_best = root_mean_squared_error(y_test, y_pred_best)
# r2 best
r2_best = r2_score(y_test, y_pred_best)
# accuracy best
accuracy_best = calculate_accuracy(y_test, y_pred_best)


# Printing Best Model Name
print(f'Best Model: {best_model.__class__.__name__}')
# Mean Absolute Error MAE
print(f'Mean Absolute Error: {mae_best}')
# Mean Absolute Error MSE
print(f'Mean Squared Error: {mse_best}')
# Mean Root Error RMSE 
print(f'Root Mean Squared Error: {rmse_best}')
# F R Squared r_2 best 
print(f'R-squared: {r2_best}')

# Accuracy Prediction Based on the Best All the Above Aspects
print(f'Prediction Accuracy: {accuracy_best:.2f}%')
