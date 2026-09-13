import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

# 1. Safe file path handling
script_dir = Path(__file__).parent 
csv_path = script_dir / 'Housing.csv'
dataset = pd.read_csv(csv_path)

# 2. Modern, clean column data mapping (replaces the deprecated .replace warning)
binary_map = {'no': 0, 'yes': 1}
dataset['driveway'] = dataset['driveway'].map(binary_map)
dataset['recroom'] = dataset['recroom'].map(binary_map)
dataset['aircon'] = dataset['aircon'].map(binary_map)
dataset['housing_type'] = dataset['housing_type'].map({'Low Cost': 0, 'Mid Cost': 1, 'High End': 2})

print(dataset.head())

# 3. Define feature columns explicitly
features = ['lotsize', 'bedrooms', 'bathrooms', 'stories', 'garage', 'driveway', 'recroom', 'aircon']

# 4. Train the model
model = LogisticRegression(max_iter=500)
model.fit(dataset[features], dataset['housing_type'])

# 5. Test input parameters
test_lotsize = 30000
test_bedrooms = 2
test_bathrooms = 2
test_stories = 1
test_garage = 2
test_driveway = 'yes'
test_recroom = 'yes'
test_aircon = 'yes'

# 6. Transform test string inputs into numerical flags
test_driveway = 1 if test_driveway == 'yes' else 0
test_recroom = 1 if test_recroom == 'yes' else 0
test_aircon = 1 if test_aircon == 'yes' else 0

test_data = [[
    test_lotsize, 
    test_bedrooms, 
    test_bathrooms,  # Fixed: used to be test_bedrooms
    test_stories, 
    test_garage, 
    test_driveway, 
    test_recroom, 
    test_aircon
]]

# 8. Predict and print outputs
output = model.predict_proba(test_data)
print("Low Cost: \t", "{:.7f}".format(output[0][0]))
print("Mid Cost: \t", "{:.7f}".format(output[0][1]))
print("High End: \t", "{:.7f}".format(output[0][2]))
