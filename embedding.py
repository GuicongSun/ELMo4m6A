import argparse
import os
import re
import sys
import numpy as np
import pandas as pd
from flair.embeddings import WordEmbeddings, ELMoEmbeddings
from flair.data import Sentence


def fa2ELMo(path1, path2):
    with open(path1) as f:
        records = f.read()
    if re.search('>', records) == None:
        print('The input RNA sequence is not fasta format.')
        sys.exit(1)
    records = records.split('>')[1:]
    PN = []
    y = []
    for fasta in records:
        line = list(fasta.split('\n'))
        seq = line[1]
        y_label = line[0][0]
        if y_label == "+":
            y.append(1)
        if y_label == "-":
            y.append(0)
        PN.append(seq)
    X = np.array(PN)
    y = np.array(y)
    X.reshape(-1, 1)
    y.reshape(-1, 1)

    # Split the gene sequence into 1-gram sequences.
    p = []
    for j in range(len(X)):
        k = ""
        for i in range(40):
            k = k + X[j][i] + " "
        k = k + X[j][40]
        p = p + [k]
    X = np.array(p)

    embedding = ELMoEmbeddings('small')
    high = 768
    PN = []
    for i in range(X.shape[0]):
        sentence = Sentence(X[i])
        embedding.embed(sentence)
        temp = []
        for token in sentence:
            swap1 = token.embedding.cpu().numpy()
            temp.append(swap1)
        high = len(temp[0])
        PN.append(temp)
        if i % 200 == 0:
            print("Coding----")

    dot_dim = 41 * high  # Word length * word embedding dimension
    PN = np.reshape(np.array(PN), (len(PN), dot_dim))
    ind_Xy = np.insert(PN, 0, values=y, axis=1)
    dataframe_ind = pd.DataFrame(ind_Xy)
    dataframe_ind.to_csv(path2, index=False, sep=',')


def main():
    parser = argparse.ArgumentParser(description="Load and preprocess data")
    parser.add_argument("--path_fa", type=str, required=True)
    parser.add_argument("--path_ELMo", type=str, required=True)
    args = parser.parse_args()

    path_fa = os.path.abspath(args.path_fa)
    path_ELMo = os.path.abspath(args.path_ELMo)

    if not os.path.exists(path_fa):
        print("The benchmark data not exist! Error\n")
        sys.exit()
    fa2ELMo(path_fa, path_ELMo)
    print('=== I get it ===')


if __name__ == "__main__":
    main()

#   h_b h_k h_l m_b m_h m_k m_l m_t r_b r_k r_l
#   python embedding.py --path_fa databases/benchmark/r_l_all.fa --path_ELMo databases/benchmark_elmo/r_l.csv
#   python embedding.py --path_fa databases/independent/r_l_Test.fa --path_ELMo databases/independent_elmo/r_l.csv




