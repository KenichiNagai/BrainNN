# https://qiita.com/pokotsun/items/dd8eb48fadeee052110b
# ゼロから作るReservoir Computing


import numpy as np
from input_generator import InputGenerator
from reservoir_computing import ReservoirNetWork
import matplotlib.pyplot as plt

# T = np.pi * 16 
# RATIO_TRAIN = 0.9
# dt = np.pi * 0.01
# AMPLITUDE = 0.9
# NUM_TIME_STEPS = int(T/dt)

LEAK_RATE=0.9
NUM_INPUT_NODES = 1
NUM_RESERVOIR_NODES = 100
NUM_OUTPUT_NODES = 1




# example of activator
def ReLU(x):
    return np.maximum(0, x)

def main():
    # i_gen = InputGenerator(0, T, NUM_TIME_STEPS)
    # data = i_gen.genetare_logistic()

    x0 = [0.1]
    x = np.array(x0)
    print(x)
    a = 3.9
    for i in range(1101):
        # print(x[-1])
        # print(a * x[-1] * (1- x[-1]))
        x = np.append(x, a * x[-1] * (1.0- x[-1]))
    data = x

    # print(data)


    num_train = 1001
    train_data = data[:num_train]

    model = ReservoirNetWork(inputs=train_data,
        num_input_nodes=NUM_INPUT_NODES, 
        num_reservoir_nodes=NUM_RESERVOIR_NODES, 
        num_output_nodes=NUM_OUTPUT_NODES, 
        leak_rate=LEAK_RATE)

    model.train() # 訓練
    train_result = model.get_train_result() # 訓練の結果を取得

    num_predict = int(len(data[num_train:]))
    predict_result = model.predict(num_predict)

    t = np.arange(1102)

    ## plot
    plt.plot(t, data, label="inputs")
    plt.plot(t[:num_train], train_result, label="trained")
    plt.plot(t[num_train:], predict_result, label="predicted")
    plt.axvline(x=1000, label="end of train", color="green") # border of train and prediction
    plt.legend()
    plt.title("ESN - Logistic Prediction")
    plt.xlabel("step")
    plt.ylabel("x_n")
    plt.show()

if __name__=="__main__":
    main()