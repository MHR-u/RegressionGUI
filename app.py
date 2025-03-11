import streamlit as st
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

def perform_regression(X, Y, degree):
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X.reshape(-1, 1))
    model = LinearRegression()
    model.fit(X_poly, Y)
    Y_pred = model.predict(X_poly)
    equation = " + ".join([f"{coef:.4f}x^{i}" for i, coef in enumerate(model.coef_)])
    equation = f"{model.intercept_:.4f} + {equation}"
    r2 = r2_score(Y, Y_pred)
    return equation, r2

st.title("Regression Model Finder")
st.write("Enter 5 pairs of data points:")

x_values = [st.number_input(f"X{i+1}", value=0.0) for i in range(5)]
y_values = [st.number_input(f"Y{i+1}", value=0.0) for i in range(5)]

degree = st.slider("Select Degree of Polynomial", 1, 4, 1)

if st.button("Find Best-Fitting Model"):
    X = np.array(x_values)
    Y = np.array(y_values)
    
    equation, r2 = perform_regression(X, Y, degree)
    
    st.subheader("Best-Fitting Equation")
    st.write(f"Equation: {equation}")
    st.write(f"R² Score: {r2:.4f}")
