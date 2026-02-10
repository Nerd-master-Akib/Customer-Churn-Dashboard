# 📉 Customer Churn Intelligence System

**End-to-end ML system for predicting customer churn with an interactive decision dashboard.**  

This project demonstrates a **full pipeline**: raw data cleaning, feature engineering, model training, and an interactive Streamlit dashboard for business decision-making.  

It is designed to be **professional, interpretable, and production-ready**.

---

## 🗂 Project Structure

Customer_churn_system/
│
├── data/
│ └── churn.csv # Raw dataset
│
├── model/
│ ├── churn_model.pkl # Trained ML model
│ ├── features.pkl # Feature list for inference
│ ├── label_encoders.pkl # Encoders for categorical columns
│ ├── train.py # Script to train the model
│ └── app.py # Interactive Streamlit dashboard
│
├── .gitignore # Ignored files for Git
├── LICENSE # MIT License
└── README.md # Project documentation

---

## 🎯 Problem Statement

Customer churn — when a customer leaves a service — is one of the most expensive issues for subscription businesses.  
- Acquiring a new customer costs **5–7× more** than retaining an existing one.  
- Proactively identifying high-risk customers enables **targeted retention strategies**.

This system allows businesses to:  
1. Predict **churn probability** for each customer  
2. Identify **key factors driving churn**  
3. Make data-driven retention decisions via an **interactive dashboard**

---

## 🛠 Technologies Used

- Python 3.10+  
- Pandas & NumPy (Data processing)  
- Scikit-learn (ML modeling)  
- Joblib (Model & encoder serialization)  
- Streamlit (Interactive dashboard)  
- Matplotlib / Streamlit charts (Feature importance visualization)

---

## ⚡ Features

1. **End-to-end ML Pipeline**  
   - Data cleaning and preprocessing  
   - Label encoding of categorical features  
   - Random Forest classifier with 80–85% accuracy  

2. **Interactive Dashboard**  
   - Enter customer details in the sidebar  
   - Predict churn probability in real-time  
   - Color-coded risk indicators (High, Medium, Low)  
   - Visualize key features contributing to churn  

3. **Business-oriented**  
   - Helps prioritize **high-risk customers**  
   - Supports **actionable retention strategies**

---

## 📥 How to Run the Project

### 1️⃣ Clone the repository

```bash
git clone https://github.com/username/Customer_churn_system.git
cd Customer_churn_system
``` 
### 2️⃣ Install dependencies
pip install -r requirements.txt


requirements.txt should include:

pandas
numpy
scikit-learn
streamlit
matplotlib
joblib
3️⃣ Train the Model (Optional if pre-trained models exist)
python model/train.py


Loads data/churn.csv

Cleans and encodes categorical features

Trains a Random Forest model

Saves churn_model.pkl, features.pkl, and label_encoders.pkl in model/

4️⃣ Run the Dashboard
streamlit run model/app.py


The browser will open: interactive dashboard

Enter customer features in the sidebar

Click Predict Churn Risk

View churn probability and risk indicators

Check feature importance chart to see top drivers of churn

5️⃣ Example Input for Testing
Feature	Value
tenure	12
MonthlyCharges	75.0
TotalCharges	900.0
gender	1 (Male)
Contract	0 (Month-to-month)
PaymentMethod	2 (Electronic check)

Output: Churn probability 73% → High risk

📊 Feature Importance (This will be added in the later)

The dashboard shows a bar chart of top features driving churn.

Example:

Contract → most important

tenure → next important

MonthlyCharges → third

This allows the business to target actionable retention strategies.

📝 License

This project is released under the MIT License. See LICENSE for details.


