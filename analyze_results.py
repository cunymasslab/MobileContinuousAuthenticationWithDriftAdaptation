import csv 
import pandas as pd


def load_results(filepath='results.csv'):
    return pd.read_csv(filepath)

df = load_results()

# summary per condition
print(df.groupby('train_condition')[['accuracy', 'f1', 'eer', 'runtime_seconds']].mean())