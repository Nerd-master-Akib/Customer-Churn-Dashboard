# -------------------------------
# 1️⃣ IMPORT LIBRARIES
# -------------------------------
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# 2️⃣ LOAD DATA

df = pd.read_csv("c:/Users/Win Technology/Downloads/archive (2)/WA_Fn-UseC_-Telco-Customer-Churn.csv")


# Convert TotalCharges to numeric, coerce errors to NaN
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Drop rows with missing values
df.dropna(inplace=True)

# -------------------------------
# 4️⃣ ENCODE CATEGORICAL FEATURES
# -------------------------------
label_encoders = {}

for col in df.select_dtypes(include='object'):
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le  # save encoder for future use
    
# 5️⃣ SPLIT FEATURES AND TARGET
X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy*100:.2f}%")

os.makedirs("model", exist_ok=True)

joblib.dump(model, "model/churn_model.pkl")
joblib.dump(X.columns.tolist(), "model/features.pkl")

# Optional: save label encoders for future use
joblib.dump(label_encoders, "model/label_encoders.pkl")

print("✅ Model, features, and encoders saved in 'model/' folder")
