from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pandas as pd
import numpy as np 

# 1. Load data
data = pd.read_csv("students.csv")

# 2. Define Features and Target
X = data[['Study_Hours']]
Y = data['Final_Score']

# 3. Train the model
model = LinearRegression()
model.fit(X, Y)

# 4. Handle User Prediction
user_input = float(input("Enter the number of hours a student study: "))
# predict is a single value based on user input
user_predict = model.predict([[user_input]])

# 5. Handle Error Metrics (Predict for the WHOLE dataset to compare sizes)
# We calculate 'all_predictions' so it has 100 values, matching Y's 100 values
all_predictions = model.predict(X)

mae = mean_absolute_error(Y, all_predictions)
mse = mean_squared_error(Y, all_predictions)
rmse = np.sqrt(mse)

# 6. Results
print(f"\nThe prediction for {user_input} hours is: {user_predict[0]:.2f}")
print("-" * 30)
print(f"Model Performance Metrics:")
print(f"MAE  : {mae:.4f}")
print(f"MSE  : {mse:.4f}")
print(f"RMSE : {rmse:.4f}")