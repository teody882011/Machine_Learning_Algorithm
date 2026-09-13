import numpy as np
from sklearn.linear_model import LinearRegression

################### Creating Dataset #############################
## House Size and Number of Rooms
X = np.array([ [2104, 3],[1600, 3],[2400, 3],[1416, 2],[3000, 4],[1985, 4],[1534, 3],[1427, 3],[1380, 3],[1494, 3],[1940, 4],[2000, 3],[1890, 3],[4478, 5],[1296, 3] ])

## Given House Price
y = [3399900,329900,369000,232000,539900,299900,314900,198999,212000,242500,239999,347000,329999,699900,259900] 
print('y', y)

################### Create and Initialize Model #############################
regModel = LinearRegression()

## Fit the model
reg = regModel.fit(X, y)

getScore = reg.score(X, y)
getCoef = reg.coef_
getIntercept = reg.intercept_

print('score', getScore)
print('getCoef', getCoef)
print('getIntercept', getIntercept)

################### Predict the test value #############################
set_HouseSize = 1000
set_Number_of_Rooms = 3
predicted_value = reg.predict(np.array([[set_HouseSize, set_Number_of_Rooms]]))
print('predicted_value', predicted_value)