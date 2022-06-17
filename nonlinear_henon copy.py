import math
import statistics
from cv2 import sqrt
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from csvFunctions import outputCSV

a = 1.0
b = 0.3
x0 = 0.1
y0 = 0.1

e0 = np.array([ [-1/math.sqrt(2)], [1/math.sqrt(2)] ])
f0 = np.array([ [ 1/math.sqrt(2)], [1/math.sqrt(2)] ])
result = []

for n in range(60):
    a = 1.0 + 0.01*n
    if a > 1.5:
        break
    
    x = [x0]
    y = [y0]
    l_sum = []
    s_sum = []

    
    for i in tqdm(range(100000)):
        x.append( 1 - a*(x[-1]**2) + y[-1])
        y.append(b * x[-1])

        df = np.array([[-2 * a * x[-1] , 1],[b, 0]])

        e1 = df @ e0
        f1 = df @ f0

        f1_dash = f1 - e1 * np.vdot(e1, f1)/((np.linalg.norm(e1))**2)

        l1 = math.sqrt(e1[0,0]**2 + e1[1,0]**2)
        s1 = abs(e1[0,0]*f1[1,0] - f1[0,0]*e1[1,0])

        e0 = e1 / np.linalg.norm(e1)
        f0 = f1_dash / np.linalg.norm(f1_dash)

        if i > 10000:
            l_sum.append(math.log(l1))
            s_sum.append(math.log(s1))

    lambda_1 = statistics.mean(l_sum)
    lambda_2 = statistics.mean(s_sum) - lambda_1

    result.append([a, lambda_1, lambda_2])

    if n % 5 == 0:
        fig2 = plt.figure()
        ax2 = fig2.add_subplot(1,1,1)
        ax2.scatter(x,y,marker=".")
        fig2.savefig("henon_sca_"+str(a)+".png", format="png", dpi=300)
        fig2.show()


print(result)
outputCSV(result, 'henon.csv')

plot_x = []
plot_y = []

for r in result:
    plot_x.append(r[0])
    plot_y.append(r[1])

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(plot_x,plot_y,marker=".")
fig.savefig("henon_lya.png", format="png", dpi=300)
fig.show()






