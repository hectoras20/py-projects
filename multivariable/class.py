# Multivariado
import pandas as pd
import numpy as np
import csv
from pathlib import Path

def build_df(file_name=str, target=str, index = None):
    base_path = Path(__file__).parent
    data_path = base_path
    path = data_path / f"{file_name}.csv"
    df = pd.read_csv(path)
    if index:
        df = df.set_index(index)
    return df

def handle_categorical(df):
    cat_cols = df.select_dtypes(include=['category', 'object']).columns
    df = pd.get_dummies(df, columns=cat_cols, dtype=int)
    if not cat_cols.empty:
       print('One-hot encoding was applied to these columns:', cat_cols)
    return df

def model_definition(df, target = str):
    y = df[target].values
    features = df[[col for col in df.columns if col not in [target]]]
    ones = np.ones((df.shape[0], 1))
    X= np.hstack((ones, features))
    betas = np.linalg.inv(np.transpose(X) @ X) @ np.transpose(X) @ y
    return betas, X

df = build_df('Data', target='Y house price of unit area', index='No')
betas, X = model_definition(df, target='Y house price of unit area')
print(X)

class multi_model:
    def __init__(self, file_name=str, target = str, index = None, decimals = 5):
        self.file_name = file_name
        self.target = target
        self.index = index
        self.decimals = decimals
        self.features = None
        self.df = None
        self.df_datatypes = None
    
    def load_df(self, handle_cat_var = True):
        self.df = build_df(file_name=self.file_name, target=self.target, index = self.index)
        self.features = [col for col in self.df.columns if col != self.target]
        self.df_datatypes = self.df.dtypes.to_dict()
        if handle_cat_var:
            handle_categorical(self.df)
            
        
        
model = multi_model(file_name='Data', target='Y house price of unit area', index='No')
model.load_df()
model.df