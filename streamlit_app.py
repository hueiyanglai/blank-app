import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Function to generate synthetic data
def generate_data(a, c, n):
    X = np.linspace(-10, 10, n)
    noise = np.random.normal(0, c, n)
    y = a * X + 50 + noise
    return X, y

# Main function to run the Streamlit app
def main():
    st.title("Linear Regression Analysis Tool")

    # Sidebar inputs
    st.sidebar.header("User Inputs")
    a = st.sidebar.slider("Slope (a)", -100.0, 100.0, 0.0)
    c = st.sidebar.slider("Noise Scale (c)", 0.0, 100.0, 10.0)
    n = st.sidebar.slider("Number of Points (n)", 10, 500, 100)

    # Generate data
    X, y = generate_data(a, c, n)

    # Prepare data for modeling
    X_reshaped = X.reshape(-1, 1)  # Reshape for sklearn
    model = LinearRegression()
    
    # Train-test split
    split_index = int(0.8 * n)
    X_train, X_test = X_reshaped[:split_index], X_reshaped[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    # Fit the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluation metrics
    mse = mean_squared_error(y_test, y_pred)
    r_squared = r2_score(y_test, y_pred)

    # Display metrics
    st.subheader("Model Performance Metrics")
    st.write(f"Mean Squared Error (MSE): {mse:.2f}")
    st.write(f"R-squared (R²): {r_squared:.2f}")

    # Plot results
    st.subheader("Actual vs. Predicted Values")
    plt.figure(figsize=(10, 6))
    plt.scatter(X_test, y_test, color='blue', label='Actual values')
    plt.scatter(X_test, y_pred, color='red', label='Predicted values', alpha=0.5)
    plt.plot(X_test, y_pred, color='red', linewidth=2)  # Regression line
    plt.title("Linear Regression")
    plt.xlabel("X")
    plt.ylabel("y")
    plt.legend()
    st.pyplot(plt)

# Run the app
if __name__ == "__main__":
    main()
