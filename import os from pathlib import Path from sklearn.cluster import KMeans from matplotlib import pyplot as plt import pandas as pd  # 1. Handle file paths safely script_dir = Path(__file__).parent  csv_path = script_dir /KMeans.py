import os
from pathlib import Path
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
import pandas as pd

# 1. Handle file paths safely
script_dir = Path(__file__).parent 
csv_path = script_dir / 'age_debt.csv'
dataset = pd.read_csv(csv_path)

k_range = range(1, 16)
sse = []
for k in k_range:
    test_km = KMeans(n_clusters=k)
    test_km.fit(dataset[['age', 'debt']])
    sse.append(test_km.inertia_)

km = KMeans(n_clusters=4)
dataset['cluster'] = km.fit_predict(dataset[['age', 'debt']])

datasetcluster1 = dataset[dataset.cluster ==0]
datasetcluster2 = dataset[dataset.cluster ==1]
datasetcluster3 = dataset[dataset.cluster ==2]

c1 = plt.scatter(datasetcluster1.age, datasetcluster1.debt, color= 'red')
c2 = plt.scatter(datasetcluster2.age, datasetcluster2.debt, color='green')
c3 = plt.scatter(datasetcluster3.age, datasetcluster3.debt, color= 'blue')
plt.legend((c1, c2, c3), ('Cluster 0', 'Cluster 1', 'Cluster 2'))
plt.xlabel('Age')
plt.ylabel('Debt')

test_age = 51
test_debt = 3100
test_cluster = km.predict([[test_age, test_debt]])
plt.scatter(test_age, test_debt, color='yellow')
print (dataset.head(5))
print('CLUSTER:', test_cluster)
plt.show()
