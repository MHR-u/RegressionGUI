import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
import io

def perform_regression(X, Y, degree):
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X.reshape(-1, 1))
    model = LinearRegression()
    model.fit(X_poly, Y)
    Y_pred = model.predict(X_poly)
    equation = " + ".join([f"{coef:.4f}x^{i}" for i, coef in enumerate(model.coef_)])
    equation = f"{model.intercept_:.4f} + {equation}"
    r2 = r2_score(Y, Y_pred)
    return equation, r2, Y_pred

st.set_page_config(page_title="Regression Model Finder", page_icon="📈", layout="centered")
st.title("📈 Regression Model Finder")
st.write("Enter **5 pairs of data points**, choose the polynomial degree, and get the **best-fitting equation & R² score**.")

# Style enhancements
st.markdown("""
    <style>
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
    }
    </style>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

# Input fields for X and Y values
with col1:
    x_values = [st.number_input(f"X{i+1}", value=0.0, key=f"x{i+1}") for i in range(5)]
with col2:
    y_values = [st.number_input(f"Y{i+1}", value=0.0, key=f"y{i+1}") for i in range(5)]

degree = st.slider("📊 Select Degree of Polynomial", 1, 4, 1)

if st.button("Find Best-Fitting Model"):
    X = np.array(x_values)
    Y = np.array(y_values)
    
    equation, r2, Y_pred = perform_regression(X, Y, degree)
    
    # Display results
    st.subheader("📋 Best-Fitting Equation")
    st.write(f"**Equation:** {equation}")
    st.write(f"**R² Score:** {r2:.4f}")
    
    # Prepare data for download
    results_df = pd.DataFrame({
        "X": X,
        "Y (Actual)": Y,
        "Y (Predicted)": Y_pred
    })
    csv = results_df.to_csv(index=False).encode('utf-8')
    
    # Download button
    st.download_button(
        label="💾 Download Results as CSV",
        data=csv,
        file_name="regression_results.csv",
        mime="text/csv"
    )
