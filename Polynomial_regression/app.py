import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("Housing.csv")

# -----------------------------
# Preprocessing
# -----------------------------
binary_cols = ['mainroad','guestroom','basement','hotwaterheating','airconditioning','prefarea']

for col in binary_cols:
    df[col] = df[col].map({'yes':1, 'no':0})

le = LabelEncoder()
df['furnishingstatus'] = le.fit_transform(df['furnishingstatus'])

# -----------------------------
# Features & Target
# -----------------------------
X = df.drop('price', axis=1)
y = df['price']

# -----------------------------
# Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------------
# Linear Regression
# -----------------------------
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

# -----------------------------
# Polynomial Regression
# -----------------------------
poly = PolynomialFeatures(degree=2)

X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

pr = LinearRegression()
pr.fit(X_train_poly, y_train)
y_pred_pr = pr.predict(X_test_poly)

# -----------------------------
# Evaluation
# -----------------------------
r2_lr = r2_score(y_test, y_pred_lr)
mse_lr = mean_squared_error(y_test, y_pred_lr)

r2_pr = r2_score(y_test, y_pred_pr)
mse_pr = mean_squared_error(y_test, y_pred_pr)

# -----------------------------
# UI STARTS
# -----------------------------
st.title("🏠 House Price Prediction App")
st.write("Using Linear vs Polynomial Regression")

# -----------------------------
# Show Metrics
# -----------------------------
st.subheader("📊 Model Performance")

col1, col2 = st.columns(2)

with col1:
    st.write("### Linear Regression")
    st.write(f"R² Score: {r2_lr:.4f}")
    st.write(f"MSE: {mse_lr:.2f}")

with col2:
    st.write("### Polynomial Regression")
    st.write(f"R² Score: {r2_pr:.4f}")
    st.write(f"MSE: {mse_pr:.2f}")

# -----------------------------
# Input Section
# -----------------------------
st.subheader("📝 Enter House Details")

area = st.number_input("Area", value=5000)
bedrooms = st.number_input("Bedrooms", value=3)
bathrooms = st.number_input("Bathrooms", value=2)
stories = st.number_input("Stories", value=2)
parking = st.number_input("Parking", value=1)

mainroad = st.selectbox("Main Road", ["yes","no"])
guestroom = st.selectbox("Guest Room", ["yes","no"])
basement = st.selectbox("Basement", ["yes","no"])
hotwater = st.selectbox("Hot Water Heating", ["yes","no"])
air = st.selectbox("Air Conditioning", ["yes","no"])
prefarea = st.selectbox("Preferred Area", ["yes","no"])
furnishing = st.selectbox("Furnishing", ["furnished","semi-furnished","unfurnished"])

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Price"):

    input_data = np.array([[area, bedrooms, bathrooms, stories,
                            {'yes':1,'no':0}[mainroad],
                            {'yes':1,'no':0}[guestroom],
                            {'yes':1,'no':0}[basement],
                            {'yes':1,'no':0}[hotwater],
                            {'yes':1,'no':0}[air],
                            parking,
                            {'yes':1,'no':0}[prefarea],
                            le.transform([furnishing])[0]]])

    # Linear Prediction
    lr_pred = lr.predict(input_data)

    # Polynomial Prediction
    input_poly = poly.transform(input_data)
    pr_pred = pr.predict(input_poly)

    st.success(f"💰 Linear Prediction: ₹ {lr_pred[0]:,.2f}")
    st.success(f"💰 Polynomial Prediction: ₹ {pr_pred[0]:,.2f}")

# -----------------------------
# Graph Comparison
# -----------------------------
st.subheader("📈 Model Comparison Graph")

# Use only 'area' for visualization
X_vis = df[['area']].values
y_vis = df['price'].values

# Linear fit
lr_vis = LinearRegression()
lr_vis.fit(X_vis, y_vis)

# Polynomial fit
poly_vis = PolynomialFeatures(degree=2)
X_vis_poly = poly_vis.fit_transform(X_vis)

pr_vis = LinearRegression()
pr_vis.fit(X_vis_poly, y_vis)

# Smooth curve
X_grid = np.linspace(min(X_vis), max(X_vis), 100).reshape(-1,1)

y_lr_grid = lr_vis.predict(X_grid)
y_pr_grid = pr_vis.predict(poly_vis.transform(X_grid))

# Plot
fig, ax = plt.subplots()

ax.scatter(X_vis, y_vis, label="Actual Data")
ax.plot(X_grid, y_lr_grid, label="Linear", linestyle='dashed')
ax.plot(X_grid, y_pr_grid, label="Polynomial")

ax.set_xlabel("Area")
ax.set_ylabel("Price")
ax.legend()

st.pyplot(fig)