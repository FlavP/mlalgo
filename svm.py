import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import preprocessing, cross_validation, neighbors, svm
plt.style.context('fivethirtyeight')
df = pd.read_csv('breast-cancer-wisconsin.txt')
df.replace('?', -99999, inplace=True)
df.drop(['id'], 1, inplace=True)
X = np.array(df.drop(['class'], 1))
y = np.array(df['class'])
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, train_size=0.2)
df = svm.SVC()
df.fit(X_train, y_train)
accuracy = df.score(X_test, y_test)
example_data = np.array([4,1,1,3,1,4,5,2,1])
example_data = example_data.reshape(1, -1)
prediction = df.predict(example_data)
print(accuracy)
print(prediction)