import pandas as pd
from sklearn.tree import DecisionTreeClassifier

smoke_detection= pd.read_csv('C:/Users/shree/Downloads/smoke_detection_iot.csv')
smd1= smoke_detection.drop(columns=['Fire Alarm'])
smd2= smoke_detection['Fire Alarm']

model= DecisionTreeClassifier()
model.fit(smd1, smd2)
predictions= model.predict([[1654750459, 14.701, 47.27, 1161, 400, 12902, 19452, 938.801, 1.62,	1.69, 11.17, 1.742,	0.039, 17128, 1]])
print(predictions)
