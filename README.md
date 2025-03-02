# Protein Stability Prediction

## Overview
This script predicts the impact of protein mutations on stability using machine learning. It leverages **MLPRegressor** and **GroupKFold cross-validation** to analyze protein mutation datasets.

## Features
- Loads protein stability data from a CSV file.
- Encodes categorical variables using one-hot encoding.
- Implements a neural network (MLPRegressor) to predict stability changes (ΔΔG).
- Uses **GroupKFold cross-validation** to prevent data leakage.
- Evaluates model performance with Pearson correlation.

## Requirements
Ensure you have the following dependencies installed:
```bash
pip install pandas numpy scikit-learn scipy torch matplotlib seaborn
```

## Running the Code
1. Place your dataset (`data.csv`) in the appropriate directory.
2. Update the `sys.path.append()` line to match the location of your protein embedding repository.
3. Run the script:
```bash
python protein_stability.py
```
4. The output will display the Pearson correlation, indicating model accuracy.
   <img width="766" alt="image" src="https://github.com/user-attachments/assets/66ed6f67-ef95-4061-b106-dc1ebcdef624" />


## Interpretation
A higher Pearson correlation suggests that the model accurately captures stability changes due to mutations. Low correlation may indicate missing features or limitations in the model.

## Possible Improvements
- Use **additional features** like sequence length, structural data, or evolutionary conservation.
- Experiment with different machine learning models (e.g., **Random Forest**, **XGBoost**).
- Increase dataset size for better generalization.

