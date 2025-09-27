import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# ----------------------------
# Load dataset
# ----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("creditcard.csv")
    return df

df = load_data()

st.title("💳 Credit Card Fraud Detection")
st.write("Interactive dashboard to explore transactions and detect fraud using Machine Learning.")

# ----------------------------
# Exploratory Data Analysis
# ----------------------------
st.header("📊 Data Exploration")

if st.checkbox("Show raw data"):
    st.write(df.head())

# Fraud distribution
fraud_count = df['Class'].value_counts()
fig1, ax1 = plt.subplots()
ax1.pie(fraud_count, labels=['Non-Fraud','Fraud'], autopct='%1.2f%%', colors=['#66b3ff','#ff6666'])
ax1.set_title("Fraud vs Non-Fraud")
st.pyplot(fig1)

# Transaction Amount distribution
fig2, ax2 = plt.subplots()
sns.histplot(df['Amount'], bins=50, kde=True, color='blue', ax=ax2)
ax2.set_title("Transaction Amount Distribution")
st.pyplot(fig2)

# ----------------------------
# Model Training
# ----------------------------
st.header("🤖 Train Model")

X = df.drop('Class', axis=1)
y = df['Class']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)

rf = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
y_proba = rf.predict_proba(X_test)[:,1]

st.subheader("Model Performance (Random Forest)")
st.text(classification_report(y_test, y_pred, digits=4))
st.write("Confusion Matrix:")
st.write(confusion_matrix(y_test, y_pred))
st.write("ROC-AUC Score:", round(roc_auc_score(y_test, y_proba),4))

# ----------------------------
# Prediction on New Data
# ----------------------------
st.header("🔮 Try a Prediction")

st.write("Enter transaction details:")

# Sidebar input
time = st.number_input("Time (seconds)", min_value=0.0, value=10000.0)
amount = st.number_input("Amount", min_value=0.0, value=100.0)

# Generate dummy PCA features (since dataset is anonymized)
v_features = [st.number_input(f"V{i}", value=0.0) for i in range(1,29)]

# Prepare input
user_data = np.array([time] + v_features + [amount]).reshape(1, -1)
user_data_scaled = scaler.transform(user_data)

# Predict
if st.button("Predict Fraud?"):
    prediction = rf.predict(user_data_scaled)[0]
    prob = rf.predict_proba(user_data_scaled)[0][1]

    if prediction == 1:
        st.error(f"⚠️ Fraudulent Transaction Detected! (Probability: {prob:.2f})")
    else:
        st.success(f"✅ Legit Transaction (Probability of Fraud: {prob:.2f})")
