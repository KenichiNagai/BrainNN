import math
from cv2 import sqrt
import numpy as np

# Henon map
a = 1.0
b = 0.3
x0 = 0.1
y0 = 0.1

x = [x0]
y = [y0]

e0 = np.array([[-1/math.sqrt(2)], [1/math.sqrt(2)]])
f0 = np.array([[ 1/math.sqrt(2)], [1/math.sqrt(2)]])
# print(e0)
# print(f0)

for n in range(10):
    a = 1.0 + 0.1*n
    if a == 1.0:
        a = 1.001
    
    for i in range(1000000):

        x.append( 1 - a*(x[-1]**2) + y[-1])
        y.append(b * x[-1])

        df = np.array([[-2 * a * x[-1] , 1],[b, 0]])
        print(df)

        e1 = df @ e0
        f1 = df @ f0
        # print(e1)
        # print(f1)

        f1_dash = 

        l1 = math.sqrt(e1[0,0] + e1[1,0])
        s1 = abs(e1[0,0]*f1[1,0] - f1[0,0]*e1[1,0])

        e0 = e1 / np.linalg.norm(e1)



