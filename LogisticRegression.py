import pandas as pd
import numpy as np
from sklearn import metrics 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv('sample2.csv')

df.head()
x = df.drop('Label',axis = 1)
y = df.Label

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=4)

logistic_regression = LogisticRegression(max_iter=100000)

logistic_regression.fit(x_train, y_train)
#Logistic_Regression(max_iter=10000)

y_pred = logistic_regression.predict(x_test)

accuracy = metrics.accuracy_score(y_test, y_pred)
accuracy_percentage = 100 * accuracy
print("Logistic Regression:")
print(accuracy_percentage)
