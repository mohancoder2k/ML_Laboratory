import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Dataset
data = {
    'ApplicantIncome': [2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500],
    'LoanAmount': [100, 120, 150, 170, 200, 220, 240, 260, 300, 320]
}

df = pd.DataFrame(data)

X = df[['ApplicantIncome']]
y = df['LoanAmount']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# Save model
joblib.dump(model, "loan_model.pkl")
print("✅ Model saved as loan_model.pkl")
