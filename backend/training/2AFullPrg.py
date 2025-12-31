import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
data = {
    'ApplicantIncome': [2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500],
    'LoanAmount': [100, 120, 150, 170, 200, 220, 240, 260, 300, 320]
}
df = pd.DataFrame(data)
X = df[['ApplicantIncome']]
y = df['LoanAmount']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Intercept:", model.intercept_)
print("Coefficient:", model.coef_)
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X, model.predict(X), color='red', label='Regression Line')
plt.title('Loan Amount Prediction using Linear Regression')
plt.xlabel('Applicant Income')
plt.ylabel('Loan Amount')
plt.legend()
plt.grid(True)
plt.show()
