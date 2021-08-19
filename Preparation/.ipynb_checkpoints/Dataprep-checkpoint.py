import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

classes = pd.read_csv('Data/elliptic_bitcoin_dataset/elliptic_txs_classes.csv')
edgelist = pd.read_csv('Data/elliptic_bitcoin_dataset/elliptic_txs_edgelist.csv')

features = pd.read_csv('Data/elliptic_bitcoin_dataset/elliptic_txs_features.csv', header=None)

classesorg = classes[(classes['class'] == '1') | (classes['class'] == '2')]

dataset = pd.merge(classesorg,features,how='inner',left_on='txId',right_on=0)

dataset = dataset.drop(columns=0)