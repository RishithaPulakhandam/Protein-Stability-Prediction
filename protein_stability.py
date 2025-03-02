import os
import sys
import pandas as pd
import numpy as np
import requests
from pprint import pprint
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GroupKFold
from scipy.stats import pearsonr
import torch

# Import data from protein sequence embedding repository
sys.path.append("/Users/rishithapulakhandam/Downloads/protein-sequence-embedding-iclr2019-master")
from src.alphabets import Uniprot21
from torch.nn.utils.rnn import PackedSequence
from src.utils import pack_sequences, unpack_sequences

# Load stability data
stability_data = pd.read_csv("/Users/rishithapulakhandam/Downloads/data.csv")

# Function to mutate protein sequence
def mutate_sequence(row):
    if not np.isnan(row.position):
        pos = int(row.position) - 1
        assert pos < len(row.sequence)
        new_sequence = list(row.sequence)
        new_sequence[pos] = row.mutation
        new_sequence = "".join(new_sequence)
        return new_sequence if row.sequence[pos] == row.wild_type else row.sequence
    return row.sequence

# Apply mutation and clean dataset
stability_data.loc[:, "sequence_mut"] = stability_data.apply(mutate_sequence, axis=1)
stability_data = stability_data.dropna(subset=['protein_name', 'ddG'])

# Prepare features and target variable
prediction_cols = ['pdb_id', 'wild_type', 'mutation']
X = pd.get_dummies(stability_data.loc[:, prediction_cols])
y = stability_data.ddG

groups = stability_data.protein_name
predictions, truths = [], []

# Group K-Fold cross-validation
for seed in [0, 1, 2]:
    for train_index, test_index in GroupKFold(7).split(X, y, groups):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        reg = MLPRegressor(hidden_layer_sizes=(50,), random_state=seed, max_iter=2000)
        reg.fit(X_train, y_train)
        
        predictions += reg.predict(X_test).tolist()
        truths += y_test.tolist()

# Output model performance
print("Test correlation: ", pearsonr(truths, predictions)[0])
