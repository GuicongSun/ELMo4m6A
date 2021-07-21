## ELMo4m6A
ELMo4m6A is a ELMo-based framework to predict the m6A sites of multiple tissues in three species.

## Dependency

- Python==3.6.8
- numpy==1.18.2
- pandas==1.0.2
- matplotlib==3.2.0
- flair==0.8.0.post1
- scikit_learn==0.24.1
- Keras==2.4.3
- tensorflow-gpu==2.2.0

## Introduction

- embedding.py: Dump fasta into a csv file through ELMo encoding.
- metrics.py: Calculate evaluation indicators and draw ROC curve     
- model_train.py: Be used to train the model    
- model_test.py: Be used to test the model  

    

## Usage

#### 1. Obtain and preprocess the datasets.

The experimental data is obtained from the following website : http://lin-group.cn/server/iRNA-m6A/

Dump fasta into a csv file through ELMo encoding.
```
python embedding.py --path_fa data_ptah --path_ELMo encoding_path
```


#### 2. model train

Model training based on benchmark datasets

```
python model_train.py --path_data benchmark_data_path  --output model_save_path
```

#### 3. model test

Model testing based on independent datasets

```
python model_test.py --model model_save_path --data independent_data_path  --output result_path
```

## Example
Use data r_l as a case to illustrate how ELMo4m6A works.
```
cd ELMo4m6A
python embedding.py --path_fa databases/benchmark/r_l_all.fa --path_ELMo databases/benchmark_elmo/r_l.csv
python embedding.py --path_fa databases/independent/r_l_Test.fa --path_ELMo databases/independent_elmo/r_l.csv

python model_train.py --path_data databases/benchmark_elmo/r_l.csv  --output result/r_l

python model_test.py --model result/r_l --data databases/independent_elmo/r_l.csv  --output result/r_l
```



