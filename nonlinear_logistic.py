# logistic map
import math
import statistics

import matplotlib.pyplot as plt

from csvFunctions import outputCSV


x0 = 0.1
x = [x0]
lyapunov = []

plot_x = []
plot_y = []
bif_x = []
bif_y = []

for n in range(500):
    a = 1.0 + 0.01*n
    if a >= 4:
        a = 3.999

    delta = []
    for i in range(100000):
        x.append(a * x[-1] * (1- x[-1]))
        if i > 10000:
            delta.append(a * ( 1 - 2*x[-1] ))
            bif_x.append(a)
            bif_y.append(x[-1])

    # print (x)
    # print (delta)

    sum = 0.0
    for d in delta:
        if d == 0:
            continue
        sum += math.log(abs(d))

    delta_mean = sum / len(delta)
    print(a, delta_mean)
    lyapunov.append([a, delta_mean])

    plot_x.append(a)
    plot_y.append(delta_mean)

    if a == 3.999:
        break
    
print(lyapunov)
# outputCSV(lyapunov, 'logistic.csv')
fig = plt.figure(figsize=(18.0, 6.0))
ax = fig.add_subplot(1,1,1)
ax.scatter(plot_x,plot_y,marker=".")
fig.savefig("logistic_lya.png", format="png", dpi=100)
fig.show()

fig2 = plt.figure(figsize=(18.0, 6.0))
ax2 = fig2.add_subplot(1,1,1)
ax2.scatter(bif_x,bif_y,marker=".")
fig2.savefig("logistic_bif.png", format="png", dpi=100)
fig2.show()
