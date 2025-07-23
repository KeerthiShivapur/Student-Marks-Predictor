import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pickle

# Load dataset
df = pd.read_csv("student_marks_data.csv")  # Ensure this file is in the same folder

# Features and Target
X = df[['Hours_Studied']]
y = df['Marks']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"✅ Model trained successfully!")
print(f"📉 Mean Squared Error: {mse:.2f}")
print(f"📈 R² Score: {r2:.2f}")

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("📦 Model saved as model.pkl")
