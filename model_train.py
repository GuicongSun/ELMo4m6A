import argparse
import os
import random
import sys
import time

import numpy as np
import pandas as pd
from keras import Input, Model
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.layers import Dense, Dropout, LSTM, Conv1D, MaxPooling1D, Bidirectional, Add, BatchNormalization, Activation
from keras.optimizers import Adam
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold

from metrics import map

BATCH_SIZE = 128

import tensorflow as tf

gpu = tf.config.experimental.list_physical_devices(device_type='GPU')
assert len(gpu) == 1
tf.config.experimental.set_memory_growth(gpu[0], True)


def shuffleData(X, y):
    index = [i for i in range(len(X))]
    random.shuffle(index)
    X = X[index]
    y = y[index]
    return X, y


def Models(drop_late=0.3, filters1=64, filters2=32, kernel_init="glorot_normal"):
    sequence_input = Input(shape=(41, 768))  # 1gram

    output = Conv1D(filters=filters1, kernel_size=2, activation='relu', padding="same", kernel_initializer=kernel_init)(
        sequence_input)
    output = BatchNormalization()(output)
    output = Activation('relu')(output)
    output1 = Dropout(drop_late)(output)

    output1 = Conv1D(filters=filters2, kernel_size=2, activation='relu', padding="same",
                     kernel_initializer=kernel_init)(output1)
    output1 = MaxPooling1D(pool_size=2)(output1)
    output1 = Dropout(drop_late)(output1)

    output1 = Bidirectional(LSTM(32, unit_forget_bias=1.2))(output1)  # ,return_sequences=True
    output1 = Dropout(0.3)(output1)

    output = Conv1D(filters=filters1, kernel_size=3, activation='relu', padding="same", kernel_initializer=kernel_init)(
        sequence_input)  # lecun_normal
    output = BatchNormalization()(output)
    output = Activation('relu')(output)
    output2 = Dropout(drop_late)(output)

    output2 = Conv1D(filters=filters2, kernel_size=2, activation='relu', padding="same",
                     kernel_initializer=kernel_init)(output2)
    output2 = MaxPooling1D(pool_size=2)(output2)
    output2 = Dropout(drop_late)(output2)

    output2 = Bidirectional(LSTM(32, unit_forget_bias=1.2))(output2)  # ,return_sequences=True
    output2 = Dropout(0.3)(output2)

    output = Add()([output1, output2])
    output = Dropout(0.3)(output)

    output = Dense(64, activation='relu')(output)
    output = Dropout(0.3)(output)
    output = Dense(1, activation='sigmoid')(output)

    model = Model(inputs=sequence_input, outputs=output)
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

    print("The model is as follows:", model)
    model.summary()

    return model


def model_train(X, y, out, batch_size=BATCH_SIZE, epochs=100):
    print('ELMo4m6A training start----')

    result_train = []
    result_verify = []

    label_encoder = preprocessing.LabelEncoder()
    y = label_encoder.fit_transform(y)

    folds = StratifiedKFold(5).split(X, y)
    for i, (trained, valided) in enumerate(folds):
        X_train, y_train = X[trained], y[trained]
        X_valid, y_valid = X[valided], y[valided]

        net = Models()

        # 3个回调函数
        best_saving = ModelCheckpoint(filepath='%s.%d.h5' % (out, i), monitor='val_loss', verbose=1,
                                      save_best_only=True)
        early_stopping = EarlyStopping(monitor='val_loss', patience=25)
        reduct_L_rate = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10)

        print("····················fold %d start····················" % (i + 1))
        net.fit(X_train, y_train, epochs=epochs, validation_data=(X_valid, y_valid), verbose=2,
                callbacks=[best_saving, early_stopping, reduct_L_rate], batch_size=batch_size)
        print("Validation test:", net.evaluate(X_valid, y_valid, batch_size=batch_size))

        result_train.append(map.calculateScore(X_train, y_train, net))
        result_verify.append(map.calculateScore(X_valid, y_valid, net))

    result_dict = (result_train, result_verify)
    map.analyze_train(result_dict, out)


def funciton(Path_data, Path_output):
    data_ben = np.array(pd.read_csv(Path_data))
    X = data_ben[:, 1:]
    y = data_ben[:, 0]
    X = np.reshape(X, (len(X), 41, 768))

    X, y = shuffleData(X, y)

    model_train(X, y, out=Path_output)
    print('=== I get it ===')


def main():
    parser = argparse.ArgumentParser(description="ELMo4m6A training----")

    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--path_data", type=str, required=True)
    args = parser.parse_args()

    Path_data = os.path.abspath(args.path_data)
    Path_output = os.path.abspath(args.output)

    if not os.path.exists(Path_output):
        os.makedirs(args.output)
    if not os.path.exists(Path_data):
        print("The benchmark data not exist! Error\n")
        sys.exit()

    funciton(Path_data, Path_output)


if __name__ == "__main__":
    ts = time.time()
    main()
    print("Training Time: ", (time.time() - ts) / 60, "minutes")

#  python model_train.py --path_data databases/benchmark_elmo/r_l.csv  --output result/r_l
