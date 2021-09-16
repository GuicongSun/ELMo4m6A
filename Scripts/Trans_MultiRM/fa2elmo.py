# Title     : fa2elmo
# Objective : TODO
# Created by: sunguicong
# Created on: 2021/7/30

import argparse
import os
import random

import torch
from flair.embeddings import WordEmbeddings, ELMoEmbeddings

from flair.data import Sentence

import re
import sys

import numpy as np
import pandas as pd

def shuffleData(X, y):
    index = [i for i in range(len(X))]
    random.shuffle(index)
    X = X[index]
    y = y[index]
    return X, y

def fa2elmo(file,path_to,gram=1):
    with open(file) as f:
        records = f.read()
    if re.search('>', records) == None:
        print('The input RNA sequence must be fasta format.')
        sys.exit(1)
    records = records.split('>')[1:]
    PN = []
    y = []
    for fasta in records:
        line=list(fasta.split('\n'))
        seq=line[1]
        y_label = line[0][0]
        if y_label == "+":
            y.append(1)
        if y_label == "-":
            y.append(0)
        PN.append(seq)
    X = np.array(PN)
    y=np.array(y)
    print(X.shape,y.shape)
    print("``~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~````~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~````~")
    X.reshape(-1, 1)
    y.reshape(-1,1)

    X, y = shuffleData(X, y)

    p=[]
    for j in range(len(X)):
        k = ""
        for i in range(100):
            k = k + X[j][i] + " "
        k = k + X[j][100]
        p=p+[k]
    X=np.array(p)


    embedding = ELMoEmbeddings('small')     #

    high = 768
    PN = []

    for i in range(X.shape[0]):
        sentence = Sentence(X[i])  # [j]
        embedding.embed(sentence)
        temp = []
        for token in sentence:
            swap1 = token.embedding.cpu().numpy()
            temp.append(swap1)
        high = len(temp[0])
        PN.append(temp)
        if i % 200 == 0:
            print("在运行----",high)


    dot_dim = 101 * high
    PN = np.reshape(np.array(PN), (len(PN), dot_dim))
    ind_Xy = np.insert(PN, 0, values=y, axis=1)
    dataframe_ind = pd.DataFrame(ind_Xy)
    dataframe_ind.to_csv(path_to, index=False, sep=',')




#   h_b     h_k h_l m_b m_h m_k m_l m_t r_b r_k r_l

def main():
    parser = argparse.ArgumentParser(description="move"+'-'*20)

    parser.add_argument("--path1", type=str,  required=True)
    parser.add_argument("--path2", type=str,  required=True)
    args = parser.parse_args()

    path1 = os.path.abspath(args.path1)
    path2 = os.path.abspath(args.path2)

    if not os.path.exists(path1):
        print("The csv benchmark_data not exist! Error\n")
        sys.exit()
    fa2elmo(path1, path2)
    print("you did it")


if __name__ == "__main__":
    main()


#   测试
#   python data2elmo.py --path1 text.fa --path2 text.csv





