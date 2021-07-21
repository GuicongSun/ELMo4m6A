import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import load_model
from metrics import map

gpu = tf.config.experimental.list_physical_devices(device_type='GPU')
assert len(gpu) == 1
tf.config.experimental.set_memory_growth(gpu[0], True)

BATCH_SIZE = 128


def model_test(model, X, y, out, batch_size=BATCH_SIZE):
    print('ELMo4m6A testing start----')
    result = []

    for i in range(5):
        net = load_model(model + '.%d.h5' % i)
        print("Independent test:", net.evaluate(X, y, batch_size=batch_size))
        print("``~~~~~~~~~````~~~~~~~~~````~~~~~~~~~````~~~~~~~~~``")
        result.append(map.calculateScore(X, y, net))

    map.analyze_test((result), out)


def funciton(model, datapath, Output):
    data_ind = np.array(pd.read_csv(datapath))
    X = data_ind[:, 1:]
    y = data_ind[:, 0]

    X = np.reshape(X, (len(X), 41, 768))

    model_test(model, X, y, out=Output)
    print('=== I get it ===')


def main():
    parser = argparse.ArgumentParser(description="ELMo4m6A testing----")

    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    model = os.path.abspath(args.model)
    data = os.path.abspath(args.data)
    output = os.path.abspath(args.output)

    if not os.path.exists(output):
        os.makedirs(args.output)
    if not os.path.exists(model):
        print("The csv model not exist! Error\n")
        sys.exit()
    if not os.path.exists(data):
        print("The csv data not exist! Error\n")
        sys.exit()

    funciton(model, data, output)


if __name__ == "__main__":
    ts = time.time()
    main()
    print("training time: ", (time.time() - ts) / 60, "minutes")

#   file                          model                                test datset             out file
#   python model_test.py --model result/r_l --data databases/independent_elmo/r_l.csv  --output result/r_l
