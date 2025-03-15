import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Set MLflow experiment
mlflow.set_experiment("Model_Tracking_Phase2")

# Load dataset
df = pd.read_csv("data/50_Startups.csv")

# Convert categorical column ("State") using one-hot encoding
df = pd.get_dummies(df, columns=["State"], drop_first=True)

# Ensure all data is numeric
df = df.apply(pd.to_numeric, errors='coerce')

# Drop any remaining NaN values
df = df.dropna()

# Verify dataset has only numerical values
if df.isnull().values.any():
    raise ValueError("Dataset contains NaN values after processing. Check data cleaning steps.")

# Split features and target
X = df.drop(columns=["Profit"]).astype(float)  # Ensure float type
Y = df["Profit"].astype(float)  # Ensure float type

# Train-Test Split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

def train_model(X_train, Y_train):
    # Modelltraining mit MLflow-Logging
    with mlflow.start_run():
        run_id = mlflow.active_run().info.run_id

        print()
        print(run_id)
        print()
        
        # Modell erstellen
        model = LinearRegression()
        model.fit(X_train, Y_train)
        
        # Vorhersagen
        Y_pred = model.predict(X_test)
        
        # Metriken berechnen
        mse = mean_squared_error(Y_test, Y_pred)
        r2 = r2_score(Y_test, Y_pred)
        
        # Logging der Parameter
        mlflow.log_param("model_type", "LinearRegression")
        
        # Logging der Metriken
        mlflow.log_metric("MSE", mse)
        mlflow.log_metric("R2_Score", r2)
        
        # Logging der Artefakte (Fehlerverteilung)
        residuals = Y_test - Y_pred
        plt.hist(residuals, bins=30)
        plt.xlabel("Residuals")
        plt.ylabel("Frequency")
        plt.title("Error Distribution")
        plt.savefig("error_distribution.png")
        mlflow.log_artifact("error_distribution.png")
        
        # Logging des Modells
        mlflow.sklearn.log_model(model, "linear_regression_model")
        
        # Berechnung der Konfidenzintervalle
        X_train_sm = sm.add_constant(X_train.values.astype(float))  # Ensure proper float conversion
        Y_train_sm = Y_train.values.astype(float)
        lr = sm.OLS(Y_train_sm, X_train_sm).fit()
        conf_interval = lr.conf_int(alpha=0.05)
        pd.DataFrame(conf_interval).to_csv("confidence_intervals.csv")
        mlflow.log_artifact("confidence_intervals.csv")
        
        # Tags setzen
        mlflow.set_tag("Algorithm", "Linear Regression")
        mlflow.set_tag("Dataset_Version", "1.0")
        
        print(f"Logged MSE: {mse}, R2: {r2}")

        return run_id
    
# Modell registrieren
run_id = train_model(X_train, Y_train)

"""
if mlflow.active_run() is None:
    run_id = train_model(X_train, Y_train)
else:
    run_id = mlflow.active_run().info.run_id

#run = mlflow.active_run()
#run_id = run.info.run_id
"""

print(run_id)

mlflow.register_model(
    f"runs:/{run_id}/linear_regression_model", "Best_Model"
)
