import numpy as np
# import matplotlib.pyplot as plt

LEANING_RATE = 0.001
CYCLE = 100

def main():
    input_x = np.array([1.0, 2.0, 3.0])
    answer_y = np.array([4.0, 7.0, 12.0])
    
    # w[n段目][隠れ層n番目への重み][入力信号n番目の重み]
    weight = np.array([[[0.1,0.2,0.3],[0.4,0.5,0.6],[0.7,0.8,0.9]], [[0.1,0.2,0.3],[0.4,0.5,0.6],[0.7,0.8,0.9]]])
    print(weight)
    
    print(input_x)
    print(weight[0])
    input = input_x
    
    history = []
    for n in range(CYCLE):
        z0, z1 = forward(input, weight)
        back(input, z0, z1, answer_y, weight)
        
        error = mean_sq_error(z1, answer_y)
        print(f'{n}回目  '+"{:.4f}".format(error))
        history.append(error)
    
    # plt.plot(history)
    # plt.show()
    
    
def sigmoid(x):
    return 1/(1+np.exp(-x))


def sigmoid_dif(x):
    return sigmoid(x)*(1-sigmoid(x))


def mean_sq_error(y, t):
    return 0.5*np.sum((y-t)**2)


def forward(input, weight):
    z0 = np.dot(input, weight[0])    
    z0 = sigmoid(z0)    
    z1 = np.dot(z0, weight[1])
    return z0, z1


def back(input, z0, output, answer, weight):
    delta2 = output - answer
    grad_w1 = z0.T.dot(delta2)

    delta1 = delta2.dot(weight[1].T) * sigmoid_dif(z0)
    grad_w0 = input.T.dot(delta1)

    weight[1] -= LEANING_RATE * grad_w1
    weight[0] -= LEANING_RATE * grad_w0
    # print(weight)


if __name__ == '__main__':
    main()