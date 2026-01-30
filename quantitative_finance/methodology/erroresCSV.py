#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 27 17:10:44 2025

@author: hectorastudillo
"""
import importlib
import pandas as pd

import market_data
importlib.reload(market_data)

def corregir_año(año):
    año = str(año)
    if len(año) == 2:
        return '20' + año  # Asumimos que son años del 2000+
    return año

def corrector(ric):
    # Traigo el data crudo
    directory = '/Users/hectorastudillo/py-proyects/Actuary_Science/projects/quantitative_finance/market_data/'
    path = directory + ric + '.csv'
    raw_data = pd.read_csv(path)
    #Empezamos con la depuración en un nuevo df
    df = pd.DataFrame()
    df['date'] = raw_data.iloc[:, 0]
    df['close'] = raw_data.iloc[:, 1]
    # Tratando los formatos 
    # DATE with the format 01.01.2025 there is not problem
    df['date'] = df['date'].astype(str).str.replace('/', '.', regex=False)
    df[['mes', 'dia', 'año']] = df['date'].str.split('.', expand=True)
    df['año'] = df['año'].apply(corregir_año) # BETTER THAN WRITING THE FOLLOWING CODE
    '''
    INSTEAD OF DO THE FOLLOWING:
    df['año'] = df['año'].astype(str).str.replace('25', '2025', regex=False)
    df['año'] = df['año'].astype(str).str.replace('24', '2024', regex=False)
    df['año'] = df['año'].astype(str).str.replace('23', '2023', regex=False)
    df['año'] = df['año'].astype(str).str.replace('202025', '2025', regex=False)
    df['año'] = df['año'].astype(str).str.replace('202024', '2024', regex=False)
    df['año'] = df['año'].astype(str).str.replace('202023', '2023', regex=False)'''
    df['date'] = df[['mes', 'dia', 'año']].astype(str).agg('.'.join, axis=1)
    return df
    
prueba = corrector('SPX')

ric = 'GCARSOA1'
df = market_data.load_ts una metrica y explica que tan bueno o imeseries(ric)
    
