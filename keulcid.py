import matplotlib.pyplot as plt
import numpy as np
from matplotlib import style
from collections import Counter
import warnings
from math import sqrt
style.use('fivethirtyeight')

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

dataset = {'r' : [[1,2], [2,3], [3,1]], 'k' : [[6,5], [7,7], [8,6]]}
test_set = [5,7]
# for i in dataset:
#     for j in dataset[i]:
#         plt.scatter(j[0], j[1], color=i, s=100)
theResult = k_nearest_neigh(dataset, test_set, 3)
print(theResult)
[[plt.scatter(j[0], j[1], color=i, s=100) for j in dataset[i]] for i in dataset]
plt.scatter(test_set[0],test_set[1], color='g', s=100)
plt.show()

