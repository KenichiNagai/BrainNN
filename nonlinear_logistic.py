# logistic map
from cmath import log
import statistics


x0 = 0.1
# e0 = 0.1 + 10 ** -8

x = [x0]
a = 1.0
lyapunov = []

for n in range(50):
    a = 1.0 + 0.1*n
    if a >= 4:
        a = 3.999

    delta = []
    for i in range(1000000):
        x.append(a * x[-1] * (1- x[-1]))
        if i > 100000:
            delta.append(a * ( -x[-1]**2 - x[-1] + 1))

    # print (x)
    # print (delta)

    sum = 0.0
    for d in delta:
        sum += log(abs(d))
        # print(abs(d))

    delta_mean = statistics.mean(delta)
    print(delta_mean)
    lyapunov.append([a, delta_mean])

    if a == 3.999:
        break
    
print(lyapunov)

