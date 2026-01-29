import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st 

import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

# PREPARING DATA
df_name = 'iris_con_genero'
df = pd.read_csv('/Users/hectorastudillo/py-proyects/machine_learning/data/' + df_name + '.csv')

# Let us remove the % symbol in the first columns
'''
df['Crecimiento_de_Ventas_YoY(%)'] = df['Crecimiento_de_Ventas_YoY(%)'].str.strip('%').astype(float)
df['Crecimiento_de_EBITDA_YoY(%)'] = df['Crecimiento_de_EBITDA_YoY(%)'].str.strip('%').astype(float)
df['Crecimiento_de_Utilidad_Neta_YoY(%)'] = df['Crecimiento_de_Utilidad_Neta_YoY(%)'].str.strip('%').astype(float)'''
df = df.dropna()
df.info()

target_column = 'Species'
X = df.drop(target_column, axis=1).values # We are not deleting this into the original dataframe, notice that we only take the values.
'''
# If we need a categorical target
df['Species'] = df['Species'].astype('category').cat.codes
target_column = 'Species'
y = df[target_column].values
'''
y = df[target_column].values 


print(y.std())


print(y.shape, X.shape)

# Ploting the original data, the axis x could be any column amd the y axis is our target column.
# THIS IS USEFULL TO FIND CORRELATIONS



df_corr = pd.DataFrame(columns=['future', 'correlation'])
# Another option...
for i in df.drop(target_column, axis=1).columns: # But you must be careful with data types of X and y.
    axis_x = i
    plt.scatter(df[i].values, y)
    plt.ylabel(target_column)
    plt.xlabel(axis_x)
    plt.show()
    # Get correlation
    slope_beta, intercept_alpha, correl_r, p_value, standard_error = st.linregress(df[i].values, y) 
    # The order of x and y does not affect the correlation coefficient (r),
    # only the slope and intercept of the regression line.
    # The correlation coefficient (r) is invariant to the order of x and y.
    df_corr.loc[len(df_corr)] = [
        i,
        correl_r]
    

# Línea de ajuste simple

X_one = X[:, 1].reshape(-1, 1)
sns.regplot(x=X_one, y=y)

# Correlación (no absoluta, contextual)
contextual_corr = np.corrcoef(X_one.ravel(), y)[0,1]
'''
We already did this step in the previous for
|r| < 0.2 → probablemente inútil
|r| > 0.5 → potencialmente útil PERO RECORDEMOS QUE ESTO NO ES ABSOLUTO
'''

# Let us work with the loss functions, in this case remember that we are dealing with a regression model
reg_all = LinearRegression()

df['Species'] = df['Species'].astype('category').cat.codes
# target_column = 'Species_code'


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42) # The argument stratify = y is not useful with regresion models, only for classification models.
y_pred_baseline = np.full_like(y_test, y_train.mean())

reg_all.fit(X_train, y_train)

y_pred_model = reg_all.predict(X_test)
	

mse_model = mean_squared_error(y_test, y_pred_model)
mse_baseline = mean_squared_error(y_test, y_pred_baseline)
rmse_model = np.sqrt(mse_model)
rmse_baseline = np.sqrt(mse_baseline)

if rmse_model >= rmse_baseline:
    print(" Modelo rechazado: no mejora al baseline")
else:
    print(" Modelo aceptable: mejora al baseline")
    print('rmse_model: ', rmse_model)
    print('rmse_baseline: ', rmse_baseline)
    print('Score: ', reg_all.score(X_test, y_test))
    print('correlations with features: ', df_corr)
    
    


## CROSS VALIDATION
# CODE TO PERFORM IT:
from sklearn.model_selection import cross_val_score, KFold

# The next line only define about the split of data, the blocks features that we want.
kf = KFold(n_splits = 6, shuffle = True, random_state = 42)
"""
El argumento shuffle en KFold (de sklearn.model_selection) controla si los datos se mezclan aleatoriamente antes de dividirse en los pliegues (folds).
Por defecto, shuffle=False, lo que significa: Los datos se dividen en el orden en que están en el dataset.

Cuando un proceso dentro de scikit-learn (o incluso en NumPy, pandas, etc.) implica algún tipo de aleatoriedad, puedes usar el argumento random_state (o seed en NumPy) para fijar esa aleatoriedad y lograr reproducibilidad.
Si un método genera resultados aleatorios, usar random_state permite que ese "azar" sea el mismo cada vez que ejecutes el código.
"""

# We could define the methods that we want to assess them and find which one is better.
model = LinearRegression()

cv_results = cross_val_score(model, X, y, cv = kf)

# With this we could even get the main, the standard desviation, quantiles, etc. with Numpy
print(cv_results)
print(np.mean(cv_results), np.std(cv_results)) ## NO IGNOREMOS LA DISPERSIÓN
#print(np.quantile(cv_results, [0.025, 0.975]))


cv_results.min()
cv_results.max()
##############################################################################################################
########## Tuning Hyperparameters




############################################################################################################################################################
#####  WORKING WITH LASSO
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge

scores = []
for alpha in [0.1, 1, 10, 100, 1000]:
    ridge = Ridge(alpha = alpha)
    ridge.fit(X_train, y_train)
    y_pred = ridge.predict(X_test)
    scores.append(ridge.score(X_test, y_test))
print(scores)

### Key: coef_ is a sensitivity metric
### Ridge with Lasso, in this case values 0 can be assigned.
target_column = 'Species'
X = df.drop(target_column, axis=1).values # We are not deleting this into the original dataframe, notice that we only take the values.
y = df[target_column].values 
names = df.drop('Species', axis = 1).columns
for i in [0.1, 1, 10, 100, 1000]:
    lasso = Lasso(alpha = i)
    lasso_coef = lasso.fit(X, y).coef_ # We take the coef attribute
    print(lasso_coef) # coef_ contiene los pesos del modelo lineal; cada valor indica cuánto cambia la variable objetivo ante un cambio unitario en la variable explicativa, manteniendo las demás constantes.
    plt.bar(names, lasso_coef)
    plt.xticks(rotation=45)
    plt.show()


