# Title     : h52fa
# Objective : TODO extracts the data in the original paper into data that I can use.
# Created by: sunguicong
# Created on: 2021/7/30
import random

import h5py
import pandas as pd
import numpy as np

def shuffleData(X):
    index = [i for i in range(len(X))]
    random.shuffle(index)
    X = X[index]
    return X

data_path='data_12RM.h5'
f = h5py.File(data_path,'r')
print(list(f.keys()))


# test_x = pd.read_hdf(data_path, 'test_in_nucleo').to_numpy()
# test_m6A=test_x[700:800,475:526]
# print(test_m6A.shape)
#
# ddd=[]
# for i in range(test_m6A.shape[0]):
#     testt = ""
#     if i<50:
#         testt+="1"
#         for j in range(51):
#             testt+=test_m6A[i][j]
#     else:
#         testt+="0"
#         for j in range(51):
#             testt+=test_m6A[i][j]
#     ddd.append(np.array(testt))
# test_m6A=np.array(ddd)
# test_m6A=shuffleData(test_m6A)
#
#
# file_test = open("test.fa","w")
# for i in range(test_m6A.shape[0]):
#     if test_m6A[i][0]=="1":
#         file_test.writelines(">+sample\n")
#     else:
#         file_test.writelines(">-sample\n")
#     file_test.writelines(test_m6A[i][1:])
#     file_test.write('\n')
# file_test.close()



# valid_x = pd.read_hdf(data_path, 'valid_in_nucleo').to_numpy()
# valid_m6A=valid_x[2100:2400,475:526]
# print(valid_m6A.shape)
#
# ddd=[]
# for i in range(valid_m6A.shape[0]):
#     testt = ""
#     if i<((valid_m6A.shape[0])/2):
#         testt+="1"
#         for j in range(51):
#             testt+=valid_m6A[i][j]
#     else:
#         testt+="0"
#         for j in range(51):
#             testt+=valid_m6A[i][j]
#     ddd.append(np.array(testt))
# valid_m6A=np.array(ddd)
#
# valid_m6A=shuffleData(valid_m6A)
#
#
# file_valid = open("valid.fa","w")
# for i in range(valid_m6A.shape[0]):
#     if valid_m6A[i][0]=="1":
#         file_valid.writelines(">+sample\n")
#     else:
#         file_valid.writelines(">-sample\n")
#     file_valid.writelines(valid_m6A[i][1:])
#     file_valid.write('\n')
# file_valid.close()




# train_x = pd.read_hdf(data_path, 'train_in_nucleo').to_numpy()
# train_m6A=train_x[58075:187946,475:526]
# print(train_m6A.shape)
#
# ddd=[]
# for i in range(train_m6A.shape[0]):
#     testt = ""
#     if i<(64972):
#         testt+="1"
#         for j in range(51):
#             testt+=train_m6A[i][j]
#     else:
#         testt+="0"
#         for j in range(51):
#             testt+=train_m6A[i][j]
#     ddd.append(np.array(testt))
# train_m6A=np.array(ddd)
#
# train_m6A=shuffleData(train_m6A)
#
#
# file_train = open("train.fa","w")
# for i in range(train_m6A.shape[0]):
#     if train_m6A[i][0]=="1":
#         file_train.writelines(">+sample\n")
#     else:
#         file_train.writelines(">-sample\n")
#     file_train.writelines(train_m6A[i][1:])
#     file_train.write('\n')
# file_train.close()









