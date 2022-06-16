import math
import statistics
from cv2 import sqrt
import numpy as np
from tqdm import tqdm

from csvFunctions import outputCSV

# Henon map
a = 1.0
b = 0.3
x0 = 0.1
y0 = 0.1

e0 = np.array([ [-1/math.sqrt(2)], [1/math.sqrt(2)] ])
f0 = np.array([ [ 1/math.sqrt(2)], [1/math.sqrt(2)] ])
# print(e0)
# print(f0)
result = []

for n in range(60):
    a = 1.0 + 0.01*n
    # if a == 1.0:
        # a = 1.001
    if a > 1.5:
        break
    
    x = [x0]
    y = [y0]
    l_sum = []
    s_sum = []
    mapping = []
    
    for i in tqdm(range(1000000)):
        # print (str(i)+'回目')

        x.append( 1 - a*(x[-1]**2) + y[-1])
        y.append(b * x[-1])
        mapping.append([x[-1], y[-1]])

        df = np.array([[-2 * a * x[-1] , 1],[b, 0]])
        # print(df)

        # print(e0)
        # print(f0)

        e1 = df @ e0
        f1 = df @ f0
        # print(e1)
        # print(f1)
        # print('inner')
        # print(np.vdot(e1, f1))

        f1_dash = f1 - e1 * np.vdot(e1, f1)/((np.linalg.norm(e1))**2)
        # print(f1_dash)

        l1 = math.sqrt(e1[0,0]**2 + e1[1,0]**2)
        s1 = abs(e1[0,0]*f1[1,0] - f1[0,0]*e1[1,0])
        # print(l1)
        # print(s1)
        # print(math.log(l1))
        # print(math.log(s1))

        e0 = e1 / np.linalg.norm(e1)
        f0 = f1_dash / np.linalg.norm(f1_dash)

        if i > 100000:
            l_sum.append(math.log(l1))
            s_sum.append(math.log(s1))

    # print(l_sum)
    # print(s_sum)

    lambda_1 = statistics.mean(l_sum)
    lambda_2 = statistics.mean(s_sum)

    # print([a, lambda_1, lambda_2])
    result.append([a, lambda_1, lambda_2])

    if n % 5 == 0:
        outputCSV(mapping, 'henonMap'+str(a)+'.csv')

print(result)
outputCSV(result, 'henon.csv')






