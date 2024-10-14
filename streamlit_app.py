# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# 1. Business Understanding: Predict house prices

# 2. Data Understanding: Generate a dataset of 100 houses
np.random.seed(42)  # For reproducibility

# Generate random data for 100 houses
sizes = np.random.randint(1000, 4000, size=100)  # House size between 1000 and 4000 sqft
bedrooms = np.random.randint(1, 6, size=100)     # Bedrooms between 1 and 5

# Add normally distributed noise with a mean of 0 and a standard deviation of 30000
noise = np.random.normal(0, 30000, size=100)

# Calculate prices with a base formula and normally distributed noise
prices = sizes * 150 + bedrooms * 10000 + noise

# Create DataFrame
df = pd.DataFrame({
    'Size': sizes,
    'Bedrooms': bedrooms,
    'Price': prices
})

# 3. Data Preparation
# Features (X) and target (y)
X = df[['Size', 'Bedrooms']]
y = df['Price']

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Modeling: Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Streamlit app starts here
st.title("House Price Prediction App")

# Display the dataset
st.write("## Dataset Preview")
st.dataframe(df)

# Input for new house features
st.sidebar.header("Input Features")
input_size = st.sidebar.number_input("Size of House (sqft)", min_value=1000, max_value=4000, value=2000)
input_bedrooms = st.sidebar.number_input("Number of Bedrooms", min_value=1, max_value=5, value=3)

# Predict for the input data
input_data = pd.DataFrame({'Size': [input_size], 'Bedrooms': [input_bedrooms]})
predicted_price = model.predict(input_data)

st.write(f"### Predicted Price for {input_bedrooms} Bedrooms and {input_size} sqft: ${predicted_price[0]:,.2f}")

# Evaluate the model and display metrics
# Fix the number of bedrooms for predictions to ensure points are on the regression line
bedrooms_constant = 3
X_test_fixed = X_test.copy()
X_test_fixed['Bedrooms'] = bedrooms_constant  # Set bedrooms to constant value

y_pred_fixed = model.predict(X_test_fixed)  # Predict using the fixed number of bedrooms

# Evaluate the model
mse = mean_squared_error(y_test, y_pred_fixed)
r2 = r2_score(y_test, y_pred_fixed)

st.write("### Model Evaluation Metrics")
st.write(f"Mean Squared Error: {mse:.2f}")
st.write(f"R-squared: {r2:.2f}")

# 6. Visualizing with testing data points and regression line
sorted_idx = X_test['Size'].argsort()
X_test_sorted = X_test.iloc[sorted_idx]
y_test_sorted = y_test.iloc[sorted_idx]
y_pred_sorted = y_pred_fixed[sorted_idx]

# Plot the actual vs predicted values for the test data
plt.figure(figsize=(10, 6))
plt.scatter(X_test_sorted['Size'], y_test_sorted, color='black', label='Actual Price (Test Points)', alpha=0.6)
plt.scatter(X_test_sorted['Size'], y_pred_sorted, color='red', label='Predicted Price (Test Points)', marker='x', alpha=0.8)

# Generate a regression line
size_range = np.linspace(X_test_sorted['Size'].min(), X_test_sorted['Size'].max(), 100)
predicted_regression_line = model.predict(pd.DataFrame({'Size': size_range, 'Bedrooms': [bedrooms_constant] * 100}))

# Plot the regression line
plt.plot(size_range, predicted_regression_line, color='blue', label='Regression Line')
plt.xlabel('Size of House (sqft)')
plt.ylabel('Price')
plt.title('House Price Prediction')
plt.legend()
plt.grid()
st.pyplot(plt)

# Run the app
if __name__ == '__main__':
    st.write("### Welcome to the House Price Prediction App!")

