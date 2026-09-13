import os
import math
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures

# ==========================================
# 1. SETUP DYNAMIC PATH & LOAD DATA
# ==========================================
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "yacht_prices_multiple_columns_poly.csv")

try:
    dataset = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"Error: Could not find '{file_path}'. Please run your CSV generator script first.")
    exit()

# ==========================================
# 2. DATA CLEANING (HANDLING MISSING VALUES)
# ==========================================
for col in ['area', 'capacity', 'rooms']:
    median_value = math.floor(dataset[col].median())
    dataset[col] = dataset[col].fillna(median_value)

features = ['area', 'capacity', 'rooms']
X = dataset[features]
y = dataset['price']

# ==========================================
# 3. TRANSFORM FEATURES TO POLYNOMIAL (DEGREE 2)
# ==========================================
# Generates columns for squared terms (x^2) and interaction terms (x1 * x2)
poly_converter = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_converter.fit_transform(X)

# ==========================================
# 4. BUILD AND TRAIN THE POLYNOMIAL MODEL
# ==========================================
poly_model = linear_model.LinearRegression()
poly_model.fit(X_poly, y)

# Extract parameters
intercept_b = poly_model.intercept_
coefficients = poly_model.coef_
feature_names = poly_converter.get_feature_names_out(features)

print("--- Polynomial Regression Model Results ---")
print(f"Intercept (b): {intercept_b:,.4f}")
print(f"coefficients (a): {coefficients[0]:,.4f}\n")

print("Polynomial Coefficients Mapping:")
for name, coef in zip(feature_names, coefficients):
    print(f"  • {name:18}: {coef:,.4f}")

# ==========================================
# 5. CALCULATE INDIVIDUAL ROW RESIDUALS (ERRORS)
# ==========================================
dataset['predicted_price'] = poly_model.predict(X_poly)
dataset['residual_error'] = dataset['price'] - dataset['predicted_price']

print("\n--- Row-by-Row Dataset Residuals (Errors) ---")
print(dataset[['price', 'predicted_price', 'residual_error']].to_string(formatters={
    'price': '{:,.2f}'.format,
    'predicted_price': '{:,.2f}'.format,
    'residual_error': '{:,.2f}'.format
}))

# Calculate global validation statistics
rss = np.sum(dataset['residual_error']**2)
mse = np.mean(dataset['residual_error']**2)
r_squared = poly_model.score(X_poly, y)

print("\n--- Global Error Metrics ---")
print(f"Residual Sum of Squares (RSS):   {rss:,.2f}")
print(f"Mean Squared Error (MSE):        {mse:,.2f}")
print(f"R-squared Accuracy Score:       {r_squared:.4f} ({r_squared*100:.2f}%)")

# ==========================================
# 6. TEST THE MODEL WITH A NEW PREDICTION
# ==========================================
test_area = 800
test_capacity = 40
test_rooms = 5

# Crucial: New inputs must go through the same polynomial transformation matrix
test_features = [[test_area, test_capacity, test_rooms]]
test_features_poly = poly_converter.transform(test_features)
predicted_price = poly_model.predict(test_features_poly)

print("\n--- Prediction Details ---")
print(f"Target Specs:    Area={test_area}, Capacity={test_capacity}, Rooms={test_rooms}")
print(f"Predicted Price: ${predicted_price[0]:,.2f}")
