# Title     : process_control
# Objective : TODO
# Created by: sunguicong
# Created on: 2021/7/30


import os

# embedding
# os.system("python fa2elmo.py --path1 databases/train1.fa --path2 databases/ELMo/train1.csv")
# os.system("python fa2elmo.py --path1 databases/train2.fa --path2 databases/ELMo/train2.csv")
# os.system("python fa2elmo.py --path1 databases/train3.fa --path2 databases/ELMo/train3.csv")
# os.system("python fa2elmo.py --path1 databases/train4.fa --path2 databases/ELMo/train4.csv")
# os.system("python fa2elmo.py --path1 databases/train5.fa --path2 databases/ELMo/train5.csv")
# os.system("python fa2elmo.py --path1 databases/train6.fa --path2 databases/ELMo/train6.csv")
# os.system("python fa2elmo.py --path1 databases/train7.fa --path2 databases/ELMo/train7.csv")

# os.system("python fa2elmo.py --path1 databases/valid.fa --path2 databases/ELMo/valid.csv")
# os.system("python fa2elmo.py --path1 databases/test.fa --path2 databases/ELMo/test.csv")




os.system("python model_MultiRM.py --data_valid databases/ELMo/valid.csv --data_test databases/ELMo/test.csv --datapath1 databases/ELMo/train1.csv --datapath2 databases/ELMo/train2.csv --datapath3 databases/ELMo/train3.csv --datapath4 databases/ELMo/train4.csv --datapath5 databases/ELMo/train5.csv --datapath6 databases/ELMo/train6.csv --datapath7 databases/ELMo/train7.csv --output result/MUltiRM")



