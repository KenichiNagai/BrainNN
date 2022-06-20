import math
import statistics
import numpy as np 
import matplotlib.pyplot as plt

from csvFunctions import outputCSV 

def henon_attractor(x, y, a, b=0.3):
	'''Computes the next step in the Henon 
	map for arguments x, y with kwargs a and
	b as constants.
	'''
	x_next = 1 - a * x ** 2 + y
	y_next = b * x
	return x_next, y_next
	
# number of iterations and array initialization
steps = 30000
X = np.zeros(steps + 1)
Y = np.zeros(steps + 1)

# starting point
X[0], Y[0] = 0, 0

e0 = np.array([ [-1/math.sqrt(2)], [1/math.sqrt(2)] ])
f0 = np.array([ [ 1/math.sqrt(2)], [1/math.sqrt(2)] ])

b = 0.3
result = []

bif_x = []
bif_y = []

for n in range(150):
    a = 0 + 0.01*n
    if a == 1.5:
        a -= 0.0001
    if a > 1.41:
        break

    l_sum = []
    s_sum = []

    X = np.zeros(steps + 1)
    Y = np.zeros(steps + 1)
    # starting point
    X[0], Y[0] = 0, 0

    # add points to array
    for i in range(steps):
        x_next, y_next = henon_attractor(X[i], Y[i], a, b)
        X[i+1] = x_next
        Y[i+1] = y_next
        if steps > 3000:
            bif_x.append(a)
            bif_y.append(x_next)

        df = np.array([[-2 * a * X[i] , 1],[b, 0]])

        e1 = df @ e0
        f1 = df @ f0

        f1_dash = f1 - e1 * np.vdot(e1, f1)/((np.linalg.norm(e1))**2)

        l1 = math.sqrt(e1[0,0]**2 + e1[1,0]**2)
        s1 = abs(e1[0,0]*f1[1,0] - f1[0,0]*e1[1,0])

        e0 = e1 / np.linalg.norm(e1)
        f0 = f1_dash / np.linalg.norm(f1_dash)

        if i > 1000:
            l_sum.append(math.log(l1))
            s_sum.append(math.log(s1))

    if n % 10 == 0:
        # plot figure
        plt.plot(X, Y, '^', alpha = 0.8, markersize=0.3)
        # plt.axis('off')
        # plt.show()
        plt.savefig("henon_sca_"+str(a)+".png", format="png", dpi=300)
        plt.close()


    lambda_1 = statistics.mean(l_sum)
    lambda_2 = statistics.mean(s_sum) - lambda_1

    print([a, lambda_1, lambda_2])
    result.append([a, lambda_1, lambda_2])

outputCSV(result, 'henon.csv')

plot_x = []
plot_y = []
for r in result:
    print(r)
    plot_x.append(r[0])
    plot_y.append(r[1])

plt.plot(plot_x, plot_y)
plt.savefig("henon_lya.png", format="png", dpi=300)
plt.close()


plot_y2 = []
for r in result:
    plot_y2.append(r[2])

plt.plot(plot_x, plot_y2)
plt.savefig("henon_lya2nd.png", format="png", dpi=300)
plt.close()


plt.plot(bif_x, bif_y, '^', alpha = 0.8, markersize=0.3)
# plt.axis('off')
# plt.show()
plt.savefig("henon_bif.png", format="png", dpi=300)
plt.close()

fig1 = plt.figure(figsize=(12.0, 6.0))
ax1 = fig1.add_subplot(1,1,1)
ax1.scatter(bif_x,bif_y,marker=".")
fig1.savefig("henon_bif_fig1.png", format="png", dpi=300)
fig1.show()

fig2 = plt.figure(figsize=(18.0, 6.0))
ax2 = fig2.add_subplot(1,1,1)
ax2.scatter(bif_x,bif_y,marker=".")
fig2.savefig("henon_bif_wide.png", format="png", dpi=300)
fig2.show()