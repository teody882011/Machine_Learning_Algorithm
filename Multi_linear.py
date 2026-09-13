import os
import math
import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Enables 3D drawing controls inside plt

# ==========================================
# 1. SETUP DYNAMIC PATH & LOAD DATA
# ==========================================
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "yacht_prices_multiple_columns.csv")

dataset = pd.read_csv(file_path)

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
# 3. BUILD AND TRAIN THE MODEL
# ==========================================
reg_model = linear_model.LinearRegression()
reg_model.fit(X, y)

# Generate predictions for all rows to plot lines
dataset['predicted_price'] = reg_model.predict(X)
slopes = reg_model.coef_
intercept_b = reg_model.intercept_

# ==========================================
# 4. GRAPH 1: SUBPLOT GRID OF ALL INDIVIDUAL DATA 
# ==========================================
# Set up a wide canvas for side-by-side trends
plt.figure(figsize=(18, 5))

colors = ['#1f77b4', '#2ca02c', '#9467bd']
titles = ['Price vs Area', 'Price vs Capacity', 'Price vs Rooms']
x_labels = ['Area', 'Capacity', 'Rooms']

for i, feature in enumerate(features):
    # Create subplot (1 row, 3 columns, position i+1)
    plt.subplot(1, 3, i + 1)
    
    # Sort data by current feature for clean line tracking
    sorted_df = dataset.sort_values(by=feature)
    
    # Plot real data points
    plt.scatter(sorted_df[feature], sorted_df['price'], color=colors[i], s=70, alpha=0.8, label='Actual Data')
    # Plot the predicted line matching your multi-variable coefficients
    plt.plot(sorted_df[feature], sorted_df['predicted_price'], color='red', linewidth=2, linestyle='--', label='Model Fit')
    
    plt.title(titles[i], fontsize=12, weight='bold')
    plt.xlabel(x_labels[i], fontsize=10)
    plt.ylabel('Price ($)', fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()

plt.suptitle('Multi-Variable Linear Regression: Individual Feature Trends', fontsize=15, weight='bold', y=1.02)
plt.tight_layout()
plt.show()  # <-- Displays the first 2D comparison matrix window

# ==========================================
# 5. GRAPH 2: 3D MODEL SPATIAL SURFACE VISUALIZATION
# ==========================================
# Instantiate a clear figure canvas specifying 3D projection framework
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Gather coordinates
x_data = dataset['area']
y_data = dataset['capacity']
z_data = dataset['price']

# Scatter real coordinates 
ax.scatter(x_data, y_data, z_data, color='blue', s=60, label='Actual Data Points')

# Create standard resolution grid tracking area and capacity spectrum boundaries
x_mesh_range = np.linspace(x_data.min(), x_data.max(), 10)
y_mesh_range = np.linspace(y_data.min(), y_data.max(), 10)
X_mesh, Y_mesh = np.meshgrid(x_mesh_range, y_mesh_range)

# Hold rooms stable at baseline middle parameter layer to map uniform plain
avg_rooms = dataset['rooms'].median()
Z_mesh = (slopes[0] * X_mesh) + (slopes[1] * Y_mesh) + (slopes[2] * avg_rooms) + intercept_b

# Drape predictive multidimensional field surface map
ax.plot_surface(X_mesh, Y_mesh, Z_mesh, alpha=0.25, cmap='Reds', edgecolor='none')

ax.set_title('Multiple Linear Regression: Price vs Area & Capacity', pad=20, fontsize=14, weight='bold')
ax.set_xlabel('Area', labelpad=10)
ax.set_ylabel('Capacity', labelpad=10)
ax.set_zlabel('Price ($)', labelpad=10)
ax.legend()

plt.show()  # <-- Displays the interactive 3D window
