import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import pickle
import json

# Load the dataset
url = "data/50_Startups.csv"
data = pd.read_csv(url)

# Preprocess the dataset
if data.isnull().sum().sum() > 0:
    data.fillna(data.median(), inplace=True)

# Convert categorical variables to numeric using one-hot encoding
if data.select_dtypes(include=['object']).shape[1] > 0:
    data = pd.get_dummies(data, drop_first=True)

# Define features and target variable
X = data.drop(columns=[data.columns[-1]])  # Assuming last column is target
Y = data[data.columns[-1]]

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=42)

# Train Linear Regression Model
model = LinearRegression()
model.fit(X_train, Y_train)

# Predictions
Y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(Y_test, Y_pred)
r2 = r2_score(Y_test, Y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared Score: {r2}")

# Save the model
with open("linear_regression_model.pkl", "wb") as f:
    pickle.dump(model, f)

# Save model coefficients for frontend usage
model_data = {
    "features": X.columns.tolist(),  # Save feature names
    "coefficients": model.coef_.tolist(),
    "intercept": model.intercept_,
    "scaler_mean": scaler.mean_.tolist(),  # Save mean values for frontend scaling
    "scaler_std": scaler.scale_.tolist()  # Save standard deviations for frontend scaling
}
with open("model_data.json", "w") as f:
    json.dump(model_data, f)

# Plot actual vs predicted values
plt.figure(figsize=(8, 6))
sns.scatterplot(x=Y_test, y=Y_pred)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted Startup Success Probability")
plt.show()
