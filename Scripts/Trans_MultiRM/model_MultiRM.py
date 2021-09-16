# Title     : model_transfer.py
# Objective : TODO
# Created by: sunguicong
# Created on: 2021/7/30



import random

import numpy as np
import os, sys, argparse
from keras import Input, Model, regularizers
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.layers import Dense, Dropout, LSTM, Conv1D, MaxPooling1D, Bidirectional, Flatten, Add, BatchNormalization, \
    Activation, Average, Concatenate
from keras.optimizers import Adam, SGD,Adadelta,RMSprop,Nadam

from metrics import map

import time


BATCH_SIZE = 128
# 32 64  128 256 512 1024

import pandas as pd

import tensorflow as tf
# gpu = tf.config.experimental.list_physical_devices(device_type='GPU')
# assert len(gpu) == 1
# tf.config.experimental.set_memory_growth(gpu[0], True)

def LSTM1(drop_late=0.3, filters1=64, filters2=32, kernel_init="glorot_normal"):
    sequence_input = Input(shape=(101, 768))  # 1gram

    kernel_size = 2

    output = Conv1D(filters=filters1, kernel_size=2, activation='relu', padding="same", kernel_initializer=kernel_init)(sequence_input)
    output = BatchNormalization()(output)
    output = Activation('relu')(output)
    output1 = Dropout(drop_late)(output)

    output1 = Conv1D(filters=filters2, kernel_size=kernel_size, activation='relu', padding="same",kernel_initializer=kernel_init)(output1)
    output1 = MaxPooling1D(pool_size=2)(output1)
    output1 = Dropout(drop_late)(output1)


    output = Conv1D(filters=filters1, kernel_size=3, activation='relu', padding="same", kernel_initializer=kernel_init)(sequence_input)  # lecun_normal
    output = BatchNormalization()(output)
    output = Activation('relu')(output)
    output2 = Dropout(drop_late)(output)

    output2 = Conv1D(filters=filters2, kernel_size=kernel_size, activation='relu', padding="same", kernel_initializer=kernel_init)(output2)
    output2 = MaxPooling1D(pool_size=2)(output2)
    output2 = Dropout(drop_late)(output2)

    output1 = Bidirectional(LSTM(32, unit_forget_bias=1.2))(output1)  # ,return_sequences=True
    output1 = Dropout(0.3)(output1)

    output2 = Bidirectional(LSTM(32, unit_forget_bias=1.2))(output2)  # ,return_sequences=True
    output2 = Dropout(0.3)(output2)

    # 这个Add()是横向叠加，前三个都是纵向叠加。
    output = Add()([output1, output2])  # average    keras.layers.Concatenate(axis=-1)
    output = Dropout(0.3)(output)


    output = Dense(64, activation='relu')(output)
    output = Dropout(0.3)(output)
    output = Dense(1, activation='sigmoid')(output)

    model = Model(inputs=sequence_input, outputs=output)
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])    #Adam(learning_rate=0.01)

    print("模型如下：", model)
    model.summary()

    return model

def generator_all(datapath1,datapath2,datapath3,datapath4,datapath5,datapath6,datapath7,batch_size):
    X_data, Y_data = [], []
    p=0 #设置读取多个文件里的数据
    while True:
        while p==0:
            file=open(datapath1, 'r')
            swap = file.readline()      # 不能删
            swap = file.readline()
            while(swap):
                #print(swap[-10:],"~~~~~~~~~~~~~")
                swap = swap.split(',')
                X = swap[1:77568]
                X.append(swap[77568][:-1])  # 去掉每一行最后的'\n'   ***********
                y = swap[0]
                for i in range(len(X)):
                    X[i] = float(X[i])
                X = np.reshape(X, (101, 768))
                X_data.append(X)
                Y_data.append(float(y))
                swap = file.readline()
                if len(X_data) == batch_size:
                    X_data = np.array(X_data)
                    Y_data = np.array(Y_data)
                    # print("抽取batch完毕。。。。")
                    yield X_data, Y_data
                    swap = file.readline()
                    X_data, Y_data = [], []
            yield X_data, Y_data
            X_data, Y_data = [], []
            p=1
        while p == 1:
            file = open(datapath2, 'r')
            swap = file.readline()  # stay
            swap = file.readline()
            while (swap):
                # print(swap[-10:],"~~~~~~~~~~~~~")
                swap = swap.split(',')
                X = swap[1:77568]
                X.append(swap[77568][:-1])  # Remove the last '\n' of each line   ***********
                y = swap[0]
                for i in range(len(X)):
                    X[i] = float(X[i])
                X = np.reshape(X, (101, 768))
                X_data.append(X)
                Y_data.append(float(y))
                swap = file.readline()
                if len(X_data) == batch_size:
                    X_data = np.array(X_data)
                    Y_data = np.array(Y_data)
                    # print("抽取batch完毕。。。。")
                    yield X_data, Y_data
                    swap = file.readline()
                    X_data, Y_data = [], []
            yield X_data, Y_data
            X_data, Y_data = [], []
            p = 2
        while p == 2:
            file = open(datapath3, 'r')
            swap = file.readline()  # 不能删
            swap = file.readline()
            while (swap):
                # print(swap[-10:],"~~~~~~~~~~~~~")
                swap = swap.split(',')
                X = swap[1:77568]
                X.append(swap[77568][:-1])
                y = swap[0]
                for i in range(len(X)):
                    X[i] = float(X[i])
                X = np.reshape(X, (101, 768))
                X_data.append(X)
                Y_data.append(float(y))
                swap = file.readline()
                if len(X_data) == batch_size:
                    X_data = np.array(X_data)
                    Y_data = np.array(Y_data)
                    # print("抽取batch完毕。。。。")
                    yield X_data, Y_data
                    swap = file.readline()
                    X_data, Y_data = [], []
            yield X_data, Y_data
            X_data, Y_data = [], []
            p = 3
        while p == 3:
            file = open(datapath4, 'r')
            swap = file.readline()  # 不能删
            swap = file.readline()
            while (swap):
                # print(swap[-10:],"~~~~~~~~~~~~~")
                swap = swap.split(',')
                X = swap[1:77568]
                X.append(swap[77568][:-1])
                y = swap[0]
                for i in range(len(X)):
                    X[i] = float(X[i])
                X = np.reshape(X, (101, 768))
                X_data.append(X)
                Y_data.append(float(y))
                swap = file.readline()
                if len(X_data) == batch_size:
                    X_data = np.array(X_data)
                    Y_data = np.array(Y_data)
                    # print("抽取batch完毕。。。。")
                    yield X_data, Y_data
                    swap = file.readline()
                    X_data, Y_data = [], []
            yield X_data, Y_data
            X_data, Y_data = [], []
            p = 4
        while p == 4:
            file = open(datapath5, 'r')
            swap = file.readline()  # 不能删
            swap = file.readline()
            while (swap):
                # print(swap[-10:],"~~~~~~~~~~~~~")
                swap = swap.split(',')
                X = swap[1:77568]
                X.append(swap[77568][:-1])
                y = swap[0]
                for i in range(len(X)):
                    X[i] = float(X[i])
                X = np.reshape(X, (101, 768))
                X_data.append(X)
                Y_data.append(float(y))
                swap = file.readline()
                if len(X_data) == batch_size:
                    X_data = np.array(X_data)
                    Y_data = np.array(Y_data)
                    # print("抽取batch完毕。。。。")
                    yield X_data, Y_data
                    swap = file.readline()
                    X_data, Y_data = [], []
            yield X_data, Y_data
            X_data, Y_data = [], []
            p = 5
        while p == 5:
            file = open(datapath6, 'r')
            swap = file.readline()  # 不能删
            swap = file.readline()
            while (swap):
                # print(swap[-10:],"~~~~~~~~~~~~~")
                swap = swap.split(',')
                X = swap[1:77568]
                X.append(swap[77568][:-1])
                y = swap[0]
                for i in range(len(X)):
                    X[i] = float(X[i])
                X = np.reshape(X, (101, 768))
                X_data.append(X)
                Y_data.append(float(y))
                swap = file.readline()
                if len(X_data) == batch_size:
                    X_data = np.array(X_data)
                    Y_data = np.array(Y_data)
                    # print("抽取batch完毕。。。。")
                    yield X_data, Y_data
                    swap = file.readline()
                    X_data, Y_data = [], []
            yield X_data, Y_data
            X_data, Y_data = [], []
            p = 6
        while p == 6:
            file = open(datapath7, 'r')
            swap = file.readline()  # 不能删
            swap = file.readline()
            while (swap):
                # print(swap[-10:],"~~~~~~~~~~~~~")
                swap = swap.split(',')
                X = swap[1:77568]
                X.append(swap[77568][:-1])
                y = swap[0]
                for i in range(len(X)):
                    X[i] = float(X[i])
                X = np.reshape(X, (101, 768))
                X_data.append(X)
                Y_data.append(float(y))
                swap = file.readline()
                if len(X_data) == batch_size:
                    X_data = np.array(X_data)
                    Y_data = np.array(Y_data)
                    # print("抽取batch完毕。。。。")
                    yield X_data, Y_data
                    swap = file.readline()
                    X_data, Y_data = [], []
            yield X_data, Y_data
            X_data, Y_data = [], []
            p = 0


def funciton(datapath1,datapath2,datapath3,datapath4,datapath5,datapath6,datapath7,data_valid,data_test ,out,batch_size=BATCH_SIZE, epochs=40):
    print('GoGoGoGoGo....Transfer Learning is ok....')

    validation_result = []
    testing_result = []
    data_valid = np.array(pd.read_csv(data_valid))
    X_val = data_valid[:, 1:]
    y_val = data_valid[:, 0]
    X_val = np.reshape(X_val, (len(X_val), 101, 768))

    data_test = np.array(pd.read_csv(data_test))
    X_test = data_test[:, 1:]
    y_test = data_test[:, 0]
    X_test = np.reshape(X_test, (len(X_test), 101, 768))
    for i in range(5):

        net = LSTM1()
        # 以下三个较为重要的函数分别起以下作用：回调以某种频率保存Keras模型或模型权重。  当监视的指标停止改进时，请停止训练。  当指标停止改进时，降低学习率。
        best_saving = ModelCheckpoint(filepath='%s.%d.h5' % (out, i), monitor='val_loss', verbose=1,
                                      save_best_only=True)  # save_best_only=True， 被监测数据的最佳模型就不会被覆盖。
        early_stopping = EarlyStopping(monitor='val_loss', patience=5)
        reduct_L_rate = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3)

        print("··································start·····································")
        print("····················fold %d start····················" % (i + 1))
        #   steps_per_epoch=（259742/2=129871）/128  = 1015   Rounded up
        net.fit_generator(generator_all(datapath1,datapath2,datapath3,datapath4,datapath5,datapath6,datapath7,batch_size),validation_data=(X_val, y_val), steps_per_epoch=1015,epochs=epochs,
                          callbacks=[best_saving, early_stopping, reduct_L_rate],
                          verbose=1,
                          )
        print("``~~~~~~~~~~~~~~~~~~Over~~~~~~~~~~~~~~~~~~~``")

        validation_result.append(map.calculateScore(X_val, y_val, net))
        testing_result.append(map.calculateScore(X_test, y_test, net))

    temp_dict = (validation_result, testing_result)
    map.analyze(temp_dict, out)




def main():
    parser = argparse.ArgumentParser(description="deep learning 6mA analysis in rice genome")

    parser.add_argument("--output", type=str, help="output folder", required=True)
    parser.add_argument("--data_valid", type=str, required=True)
    parser.add_argument("--data_test", type=str,  required=True)
    parser.add_argument("--datapath1", type=str, required=True)
    parser.add_argument("--datapath2", type=str, required=True)
    parser.add_argument("--datapath3", type=str, required=True)
    parser.add_argument("--datapath4", type=str, required=True)
    parser.add_argument("--datapath5", type=str, required=True)
    parser.add_argument("--datapath6", type=str, required=True)
    parser.add_argument("--datapath7", type=str, required=True)
    args = parser.parse_args()

    Data_valid = os.path.abspath(args.data_valid)
    Data_test = os.path.abspath(args.data_test)
    DataCSV1 = os.path.abspath(args.datapath1)
    DataCSV2 = os.path.abspath(args.datapath2)
    DataCSV3 = os.path.abspath(args.datapath3)
    DataCSV4 = os.path.abspath(args.datapath4)
    DataCSV5 = os.path.abspath(args.datapath5)
    DataCSV6 = os.path.abspath(args.datapath6)
    DataCSV7 = os.path.abspath(args.datapath7)
    OutputDir = os.path.abspath(args.output)

    if not os.path.exists(OutputDir):
        os.makedirs(args.output)
    if not os.path.exists(DataCSV1):
        print("The csv benchmark_data not exist! Error\n")
        sys.exit()
    funciton(DataCSV1,DataCSV2,DataCSV3,DataCSV4,DataCSV5,DataCSV6,DataCSV7,Data_valid,Data_test, OutputDir)

if __name__ == "__main__":
    ts = time.time()
    main()
    print("training time: ", (time.time() - ts) / 60, "minutes")




#  python model_MultiRM.py --data_valid databases/ELMo/valid.csv --data_test databases/ELMo/test.csv --datapath1 databases/ELMo/train1.csv --datapath2 databases/ELMo/train2.csv --datapath3 databases/ELMo/train3.csv --datapath4 databases/ELMo/train4.csv --datapath5 databases/ELMo/train5.csv --datapath6 databases/ELMo/train6.csv --datapath7 databases/ELMo/train7.csv --output result/MUltiRM


