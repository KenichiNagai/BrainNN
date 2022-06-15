# logistic map
import math
import statistics

from csvFunctions import outputCSV


x0 = 0.1
x = [x0]
lyapunov = []

for n in range(500):
    a = 1.0 + 0.01*n + 0.000000001
    if a >= 4:
        a = 3.999

    delta = []
    for i in range(1000000):
        x.append(a * x[-1] * (1- x[-1]))
        if i > 100000:
            delta.append(a * ( 1 - 2*x[-1] ))

    # print (x)
    # print (delta)

    sum = 0.0
    for d in delta:
        sum += math.log(abs(d))


    delta_mean = sum / len(delta)
    print(a, delta_mean)
    lyapunov.append([a, delta_mean])

    if a == 3.999:
        break
    
print(lyapunov)
outputCSV(lyapunov, 'logistic.csv')

