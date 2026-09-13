import os
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model

# 1. Load CSV safely
script_dir = Path(__file__).parent 
csv_path = script_dir / 'Housing2.csv'
df = pd.read_csv(csv_path)
print("Columns found in your CSV:", df.columns.tolist())  

# 2. Extract and reshape features
Y = df['price'].values.reshape(-1, 1)
X = df['lotsize'].values.reshape(-1, 1)

# 3. Split the data into training/testing sets
X_train = X[:-250]
X_test = X[-250:]

Y_train = Y[:-250]
Y_test = Y[-250:]

# 4. Create and train the linear regression model
regr = linear_model.LinearRegression()
regr.fit(X_train, Y_train)

# 5. Predictions (FIXED: Passed as 2D array and extracted scalar value)
prediction = regr.predict([[5000]])
print('Predicted value: ' + str(round(prediction[0][0])))

# 6. Plot outputs together (FIXED: Plotting everything sequentially before plt.show())
plt.figure(figsize=(8, 5))
plt.scatter(X_test, Y_test, color='black', alpha=0.5, label='Actual Data')
plt.plot(X_test, regr.predict(X_test), color='red', linewidth=3, label='Regression Line')

# 7. Add readable titles and labels (FIXED: Removed tick suppressors so numbers show up)
plt.title('Housing Price vs. Lot Size (Test Data)')
plt.xlabel('Lot Size')
plt.ylabel('Price')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()
