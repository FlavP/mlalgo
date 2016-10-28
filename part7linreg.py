from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import random

style.use('fivethirtyeight')
#dtype for later use where the data type matters
#xs = np.array([1, 2, 3, 4, 5, 6], dtype=np.float64)
#ys = np.array([5, 4, 6, 5, 6, 7], dtype=np.float64)

# plt.scatter(xs, ys)
# plt.show()

def create_dataset(hm, variance, step=2, correlation=False):
    val = 1
    ys = []
    for h in range(hm):
        y = val + random.randrange(-variance, variance)
        ys.append(y)
        if(correlation and correlation == "pos"):
            val += step
        elif(correlation and correlation =="neg"):
            val -= step
    xs = [i for i in range(len(ys))]
    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)

def best_fit_slope_and_intercept(xs, ys):
    m = ( (mean(xs) * mean(ys) - mean(xs * ys)) /
          (mean(xs)**2 - mean(xs**2)))
    b = mean(ys) - m * mean(xs)
    return m, b

def squared_error(ys_orig, ys_line):
    return sum((ys_orig - ys_line)**2)

def coef_func(ys_orig, ys_line):
    mean_line = [mean(ys_orig) for y in ys_orig]
    #numarator
    squared_error_regression = squared_error(ys_orig, ys_line)
    #numitor
    squared_error_y_mean = squared_error(ys_orig, mean_line)
    return 1 - squared_error_regression/squared_error_y_mean

xs, ys = create_dataset(40, 40, 2, correlation="neg")
m, b = best_fit_slope_and_intercept(xs, ys)
#print(m, b)
#regression line
bees = np.array([2, 4, 3], dtype=np.float64)
#equ = [(x**3 * bees[0] + x**2 * bees[1] - bees[1]) for x in xs]
predict_x = 8
predict_y = (m * predict_x) + b
regression_line = [(m * x) + b for x in xs]
r_squared = coef_func(ys, regression_line)
plt.scatter(xs, ys)
#plt.scatter(predict_x, predict_y, color="green")
plt.plot(regression_line)
#plt.plot(equ)
plt.show()
print(r_squared)