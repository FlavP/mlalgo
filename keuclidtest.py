import numpy as np
from collections import Counter
import pandas as pd
import warnings
from math import sqrt
import random

def k_nearest_neigh(data, predict, k=3):
    if len(data) >= k:
        warnings.warn("This is not goood")
    distances = []
    for group in data:
        for features in data[group]:
            #euclidian_distance = sqrt((data[0] - predict[0])**2 + (data[1] - predict[1])**2)
            #euclidian_distance = np.sqrt(np.sum((np.array(features) - np.array(predict))**2))
            euclidian_distance = np.linalg.norm(np.array(features) - np.array(predict))
            distances.append([euclidian_distance, group])
    k_votes = [i[1] for i in sorted(distances)[:k]]
    vote_result = Counter(k_votes).most_common(1)[0][0]
    return vote_result

df = pd.read_csv('breast-cancer-wisconsin.txt')
df.replace('?', -99999, inplace=True)
df.drop(['id'], 1, inplace=True)
full_data = df.astype(float).values.tolist()
random.shuffle(full_data)
train_set = {2 : [], 4 : []}
test_set = {2 : [], 4 : []}
test_size = 0.2
train_data = full_data[:-int(test_size * len(full_data))]
test_data = full_data[-int(test_size * len(full_data)):]

for i in train_data:
    train_set[i[-1]].append(i[:-1])
for i in test_data:
    test_set[i[-1]].append(i[:-1])

correct = 0
total = 0

for group in test_set:
    for data in test_set[group]:
        vote = k_nearest_neigh(train_set, data, k=5)
        if vote == group:
            correct += 1
        total += 1

print('Accuracy:', correct/total)

