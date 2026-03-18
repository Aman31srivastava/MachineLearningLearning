import streamlit as st
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import pandas as pd

#title

st.title("🏠 House Price Prediction (Advanced ML)")
st.write("Predict house price using multiple factors")

# Load dataset
df = pd.read_csv("Housing.csv")




# -----------------------------
# Data Preprocessing
# -----------------------------

# Convert yes/no to 1/0
binary_cols = ['mainroad','guestroom','basement',
               'hotwaterheating','airconditioning','prefarea']

for col in binary_cols:
    df[col] = df[col].map({'yes':1, 'no':0})


# One-hot encoding
df = pd.get_dummies(df, columns=['furnishingstatus'], drop_first=True)



# -------------------------------
# Features & Target
# -------------------------------
X = df.drop('price', axis=1)
y = df['price']

# -------------------------------
# Train Model
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


model = LinearRegression()
model.fit(X_train, y_train)

# Accuracy
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)

st.subheader(f"📊 Model Accuracy (R² Score): {round(r2, 2)}")

# -------------------------------
# USER INPUT UI
# -------------------------------
st.subheader("📥 Enter House Details")


area = st.number_input("Area (sq ft)", min_value=500)
bedrooms = st.selectbox("Bedrooms", [1,2,3,4,5])
bathrooms = st.selectbox("Bathrooms", [1,2,3,4])
stories = st.selectbox("Stories", [1,2,3,4])
parking = st.selectbox("Parking Spaces", [0,1,2,3])

mainroad = st.selectbox("Main Road Access", ["yes","no"])
guestroom = st.selectbox("Guest Room", ["yes","no"])
basement = st.selectbox("Basement", ["yes","no"])
hotwater = st.selectbox("Hot Water Heating", ["yes","no"])
aircon = st.selectbox("Air Conditioning", ["yes","no"])
prefarea = st.selectbox("Preferred Area", ["yes","no"])

furnishing = st.selectbox("Furnishing Status", 
                         ["furnished", "semi-furnished", "unfurnished"])

# Convert inputs
def convert(val):
    return 1 if val == "yes" else 0

mainroad = convert(mainroad)
guestroom = convert(guestroom)
basement = convert(basement)
hotwater = convert(hotwater)
aircon = convert(aircon)
prefarea = convert(prefarea)


# Furnishing encoding
furnished = 1 if furnishing == "furnished" else 0
semi_furnished = 1 if furnishing == "semi-furnished" else 0

# -------------------------------
# Prediction
# -------------------------------
if st.button("💰 Predict Price"):

    input_data = np.array([[
        area, bedrooms, bathrooms, stories, parking,
        mainroad, guestroom, basement,
        hotwater, aircon, prefarea,
        semi_furnished, furnished
    ]])

    prediction = model.predict(input_data)

    st.success(f"🏷 Estimated House Price: ₹ {int(prediction[0])}")