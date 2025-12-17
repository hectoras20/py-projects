import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import importlib
import scipy.stats as st
import os

import market_data
importlib.reload(market_data)

'''file_name = 'GCARSOA1.csv'
ric = file_name.split('.')[0]
directory = '/Users/hectorastudillo/py-proyects/Actuary_Science/projects/quantitative_finance/market_data/'
path = directory + ric + '.csv'
raw_data = pd.read_csv(path)
t = pd. DataFrame ()
t['date'] = pd.to_datetime(raw_data['Date'], dayfirst=True)
t['close'] =raw_data['Close']
t = t.sort_values(by='date', ascending=True)
t['close_previous'] = t['close'].shift(1) # This function shift "recorrer" one cell. 
t['return_close'] = t['close'] / t['close_previous'] - 1
t = t.dropna()
t = t.reset_index(drop=True)
print(t)

plt.figure()
t.plot(kind ='line', x='date', y = 'close', grid = True, title='Timeseries of close prices for' + ric)
plt.show()'''
ric = 'NVDA'
sim = market_data.distribution_manager(ric)
sim.load_timeseries()
sim.plot_timeseries()
sim.compute_stats()
sim.plot()


'''directory = '/Users/hectorastudillo/py-proyects/Actuary_Science/projects/quantitative_finance/market_data/'

rics = []
is_normals = []
for file_name in os.listdir(directory):
    print('file name=' + file_name)
    ric = file_name.split('.')[0]
    dist = market_data.distribution_manager(ric)
    dist.load_timeseries()
    dist.compute_stats()
    # generate
    rics.append(ric)
    is_normals.append(dist.is_normal)
    
df= pd.DataFrame()
df['ric'] = rics
df['is normal']= is_normals
df = df.sort_values(by='is normal', ascending=False)'''
    