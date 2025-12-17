# RECAP of sublibraries
# linear_model - Selection of LinearRegression Model
# model_selection - Some useful tools to work with the model selected (split and cross)
# metrics - So far, contains loss functions, for metrics as .score do not need this sublibrary, but in general as its name sugggests, it serves to assess.

# We will work with ... and ... datasets
"""
In this case our target will be the film´s genre column, 
the features are the remaining values, so we may delete the target column to keep the features.
Remember that we must delegate the name of the column, when we use the method drop 
without arguments, the name is still in the Serie, we delete it using the method values()
"""
import pandas as pd
import numpy as np
# PREPARING DATA
df_name = ''
df = pd.read_csv('/Users/hectorastudillo/py-proyects/machine_learning/data/' + df_name + '.csv')
df.info()

target_column = ''
X = df.drop(target_column, axis=1).values # We are not deleting this into the original dataframe
y = df[target_column].values 

print(type(X), type(y))
# Both must be numpy arrays. NO SERIES!

""" 
General Considerations:  
    - Our X and y variables must be numpy arrays. NO SERIES!!!!!!!!
    
    - The features shape must have a value (any) in its second entry, for example (x, 1), being a two dimensional array in other words.
        Solution: If it does not happen, we use the .reshape function with -1, 1 argument
            - values() method returns a list, an array, a numpy array with the values: numpy.ndarray
            - If we do not specify the method values(), we get a pandas.core.series.Series
            
        (x, 1) = 2 dimensional = (1, x)
        
        The total number of elements must match with the multiplication of the arguments.
        
        What the arguments means:
        The -1 argument compute automatically the first dimension.
        -1 = rows and 1 = columns
        
        How it works:
        (x, ) is equal to a numpy array in other words, of 1 dimension = [1,2,3]
        Instead of (x,1), which is a 2D array = [[1],[2],[3],[4]] - Each bracket inddicates a row!

"""
print(y.shape, X.shape)


# Ploting the original data, the axis x could be any column amd the y axis is our target column.
# THIS IS USEFULL TO FIND CORRELATIONS

import matplotlib.pyplot as plt
axis_x = '' # FROM FEATURES VARIABLE
plt.scatter(df[axis_x].values, y)
plt.ylabel(target_column)
plt.xlabel(axis_x)
plt.show()

# Another option...
for i in df.columns:
    axis_x = i
    plt.scatter(df[i].values, y)
    plt.ylabel(target_column)
    plt.xlabel(axis_x)
    plt.show()
    
# DATA ALREADY PREPARED

"""
Regression Models
- Binary, yes or no, 1 or 0 IS NOT A REGRESSION MODEL, IS A CLASSIFICATION MODEL.
    * from sklearn.linear_model import LogisticRegression
    * Convierte cualquier número real en un valor entre 0 y 1 - interpretable como probabilidad.
    * Finalmente, clasifica según un umbral (por defecto 0.5)
- Linear Regresion
    * Te dará valores continuos entre 0 y 1 (a veces incluso menores a 0 o mayores a 1).
    Esto puede devolver 0.3, 2.7, −1.2, etc.
    * No interpreta el resultado como una probabilidad real.
"""
########################################
"""
LINEAR REGRESSION (there exist types of it)
In general about it...

This will give us a line of best fit for our data
The line represents the linear regression model's fit of target_column values 
against the feature, which will indicate us correlation.

We could plot this line with plot or scatter function, will be the same outcome.

About the number of features that could recieve to fit and predict:
    LinearRegression ajusta un modelo del tipo:
    y^=β0+β1 x1+β2 x2+ ... +βn xn
    But we talk about it in the next section
 
    Por eso puedes pasarle una o muchas columnas,
    ya que él estima un coeficiente (βi) por cada una.
"""
from sklearn.linear_model import LinearRegression
model = LinearRegression
# CODE - WORKING WITH ONE FEATURE
X_one = X[:, 1]
model.fit(X_one, y)
# We could use the same X variable if we do not have more new data to predict 
X_new = X_one
predictions = model.predict(X_new)

# Ploting the outcomes
plt.scatter(X_one, y)
plt.plot(X_one, predictions) # This will be the line of best fit because we are using the predictions obtained previously.
plt.ylabel(axis_x)
plt.xlabel(target_column)
plt.show()

'''
SIMPLE LINEAR REGRESSION MODEL
Key:
    There are types of linear regressions
    
One future:
    Using a single feature is known as simple linear regression, where y is the target, x is the feature, and a and b are the model parameters that we want to learn. 

KEY:
    a and b are the model parameters that we want to learn. 
    a and b are also called the model coefficients, or the slope and intercept, respectively.
    y^ = ax + b

New concept:
    So, how do we accurately choose values for a and b?
    ERROR FUNCTIONS = LOST FUNCTION = COST FUNCTIONS
    Minimize Problem
    We can define an error function for any given line and then choose the line that minimizes this function. Error functions are also called loss or cost functions.

    - We want to minimize the vertical distance between the fit (regression line) and the data. 
    - This distance is called a residual. 

Linear Regressin called OLS
    This type of linear regression is called Ordinary Least Squares, or OLS, where we aim to minimize the RSS.

RSS = The residual sum of squares - Lost function since we want to minimize it:
    KEY: It does not work for just one feature variable!
    
    We square the residuals, then of this, by adding all the squared residuals, we calculate the residual sum of squares, or RSS. 

ALSO KNOWN AS MSE UNDIVIDED BY n

'''

'''
Two features:
What if I want to fit and predict with too many features?
We previously take only one featues from our features to create our model.
    
When we have two features, x1, and x2, and one target…
    There is a difference between Statsmodel and Scikit, the way to fit a model is different here, while with Statsmodel we do not need to pass arguments…
    To fit a regression model with Scikit of 2 features, we get 3 variables…
    - coefficient for feature 1 = a_1
    - coefficient for feature 2 = a_2
    - intercept = b

'''

'''
MULTIPLE LINEAR REGRESSION MODEL:
    When adding more features, it is known as multiple linear regression. Fitting a multiple linear regression model means specifying a coefficient, a n, for n number of features, and b. 
    For multiple linear regression models, scikit-learn expects one variable each for feature and target values, whose values the model fits for you and gives you.
'''

# CODE, NOW LET'S USE ALL OUR FEATURES
# It will be the same as we did previously with one variable, THE SAME, but let us split our data now.
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42, stratify = y)

reg_all = LinearRegression()
reg_all.fit(X_train, y_train)
y_pred  = reg_all.predict(X_test) 

# WHAT ABOUT y_test
# Remember that we use y_test with metric and loss functions, to see how well the model made its predictions.

"""
ABOUT THE RSS LOSS FUNCTION (there is not code about it yet)
RSS with many variables:
    Sum_{i=1}^{n} (y_i - (b + a_1 * x_1 + ... + a_n * x_n)) ^ 2
        where y_i belongs to the original data, data we are working ON.
    Where y^ = predictions =  b + a_1 * x_1 + ... + a_n * x_n
    Or b + a_1 * x_1 in simple linear regression
"""

# HOW CAN I MEASURE/EXPLAIN THE TARGET´S VARIANCE?
# We will use the same function that we used in KNN 

print(reg_all.score(X_test, y_test))
"""
.SCORE - This is not a loss function of course:
    REMEMBER THAT IT DOES NOT HAVE THE SAME FORMULA IN REGRESSION WITH KNN MODELS FOR EXAMPLE

What it did?
y_pred = knn.predict(X_test)
R^2 or mean, depends on the model

IN THIS CASE (AND ALL THE Linear Regression CASES (even fon Ridge since it is a regression model)):
    The default metric for linear regression is R-squared, which quantifies the amount of variance in the target variable that is explained by the features. 
    
    
R^2 IS NOT EQUALS RSS

R^2 = 1 - (RSS/TSS)
"""

'''
MORE LOSS FUNCTIONS FOR LINEAR REGRESSION MODELS:
   
MSE and RMSE 
DIFFERENCES:
    UNITS returned
    - MSE - Mean Squared Error (measured in target units, SQUARED) 
    - RMSE (measured in the same units ot the target variable)

MSE:
    The same as RSS but divided by n (number of features)
    In other words is the MEAN of RSS

RMSE:
    squared root of MSE
    

MSE is used to optimize but RMSE not at all.
'''

# CODE:
from sklearn.metrics import mean_squared_eror #MSE
# Remember that it is a loss function, so we pass it as arguments the y_predicted and y_test
mse = mean_squared_eror(y_test, y_pred, squared = False)
# As you noticed, we could use the same function to compute the RMSE, we only need to change the last argument to True.
rmse = mean_squared_eror(y_test, y_pred, squared = True)

###################################
# CROSS VALIDATION - Dealing with assessment dependency in a split set.
###################################
'''
There is a potential pitfall in the process of train-test split and computing model performance metrics on our test set.
Motivation 
If we're computing R-squared on our test set (i.e. we use the .score function with a Linear Reg model), 
the R-squared returned is dependent on the way that we split up the data! 

The data points in the test set may have some peculiarities that mean the R-squared computed on it IS NOT REPRESENTATIVE
of the model's ability to generalize to unseen data.

To combat this dependence on what is essentially a random split, we use a technique called cross-validation.

KEYS:
    Allow us to compare differente machine learning methods annd get a sense of how well they will work in practice
    Starts dividing the data, then uses one block to testing and the remaining to traing, then it uses the next block and the remaining to testing and so on.
    In the end every block of data is used for testing and we can compare methofd by seeing how well they permormed.
    
FOUR FOLD CROSS VALIDATION
We divided the data into 4 blocks

LEAVE ONE OUT CROSS VALIDATION 
In a extreme case, we could call each individual sample a block!

TEN FOLD CROSS VALIDATION
In practice it is very common to divide the data into 10 blocks
'''
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

"""
We then call cross_val_score, passing the model, the feature data, and the target data as the first three positional arguments. 
We also specify the number of folds by setting the keyword argument cv equal to our kf variable. 
This returns an array of cross-validation scores, which we assign to cv_results. 
"""
cv_results = cross_val_score(model, X, y, cv = kf)

# With this we could even get the main, the standard desviation, quantiles, etc. with Numpy
print(cv_results)
print(np.mean(cv_results), np.std(cv_results)) 
print(np.quantile(cv_results, [0.025, 0.975]))


###################################
# Regularized Regression - A technique to avoid overfiting
# MODELS
# RIDGE MODEL
###################################
'''
Recall that fitting a linear regression model minimizes a loss function to choose a coefficient, a, for each feature, and the intercept, b. 
If we allow these coefficients to be very large, we can get overfitting. 
Therefore, it is common practice to ALTER the loss function so that it penalizes large coefficients. 
This is called regularization.

REGULARIZATION = ALTER THE LOSS FUNCTION TO PENALIZE LARGE COEFFICIENTS.

RIDGE REGRESSION 
    OLS LOSS FUNCTION + PENALIZATION = RSS + PENALIZATION
    (Remember that OLS only minimize RSS)
    We use the Ordinary Least Squared loss function plus the squared value of each coefficient, multiplied by a constant alpha
    
    PENALIZATION =  the squared value of each coefficient, multiplied by a constant alpha

How it works?
    With Ridge we need to choose the alpha value in order to fit and predict.

    Essentially, we can select the alpha for which our model performs best.

Key: Picking alpha for ridge is similar to picking k in KNN

About this ALPHA:
    DEF. Hyperparameter = Alpha in ridge is known as a hyperparameter, which is a variable used for selecting a model´s parameters.
    
    Alpha controls model complexity
    
    When alpha equals zero, we are only performing OLS, where large coefficients are not penalized and overfitting may occur. 
    RSS + 0 = There is not a penalization, overfitting MAY occur.
    
    * A high alpha means that large coefficients are significantly penalized, which can lead to underfitting.
'''
#Code
from sklearn.linear_model import Ridge 
"""
About RIDGE model
Remember that Ridge or Ridge Regression IS A MODEL as OLS, OLS is a model only defined by minimize RSS while Ridge has a modified loss function. Each model has it loss function

Ridge es una variante de OLS cuya función de pérdida fue modificada añadiendo una penalización L2.

Cuando hablamos de penalización L2 nos referimos directamente a la norma L2 (o norma euclidiana) aplicada a los coeficientes del modelo.

En Ridge no usamos simplemente la norma L2
Usamos la normal L2 al cuadrado : ||β||_2^2 = (the square root of the sum of the squared betas) squared = the sum of the squared betas

!!!En todos los modelos regresivos de scikit-learn (incluyendo Ridge),
.score() devuelve:
R² (coeficiente de determinación)
R^2 = 1 - ((sum (y_i-y_pred)^2 ) / ())

LASSO uses the norm L1 as penalization
"""

scores = []

for alpha in [0.1, 1.0, 10.0, 100.0, 1000.0]:
    model = Ridge(alpha=alpha)
    # In the following line of code the penality is applied (in the training)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    scores.append(model.score(X_test, y_test))
    # .scores in this case, Ridge AND IN ALL THE REGRESSIVE MODELS it computes R^2
print(scores) 

###################################
# Regularized Regression - A technique to avoid overfiting
# MODELS
# LASSO MODEL
###################################
"""
Lasso Regression - Regression Model
There is another regularized regression model called lasso 
where our loss function is again the OLS loss function plus the absolute value of each coefficient 
multiplied by some constant, alpha.

The penalization is the norml L1, multiplied by some alpha
"""
# Code
from sklearn.linear_model import Lasso
scores = []

for alpha in [0.1, 1.0, 10.0, 20.0, 50.0]:
    model = Lasso(alpha=alpha)
    # In the following line of code the penality is applied (in the training)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    scores.append(model.score(X_test, y_test))
    # .scores in this case, LASSO AND IN ALL THE REGRESSIVE MODELS it computes R^2
print(scores)
"""
BUT
There is one more function with Lasso

Lasso for feature selection, the most important predictor for our target variable.

    Lasso regression can actually be used to assess feature importance
    
    This is because it tends to shrink the coefficients of less important features to zero. 
    
    The features whose coefficients are NOT SHRINK (encoger) to zero are selected by the lasso algorithm. 

    This type of feature selection is very important because it allows us to communicate results to non-technical audiences. 

    It is also useful for identifying which factors are important predictors for various physical phenomena.
"""
from sklearn.linear_model import Lasso
X = df.drop('column', axis=1).values
y = df['column'].values
column_names = df.drop('column', axis=1).columns

model = Lasso(alpha=0.1)
# best coeff = FITTING
lasso_coef = model.fit(X, y).coef_

# Ploting outcomes
plt.bar(column_names, lasso_coef)
plt.sticks(rotation=45)
plt.show()

###################################
# How good is the model?
# Recall that we can use accuracy, the fraction of correctly classified labels, to measure model performance. However, accuracy is not always a useful metric.
# CLASS IMBALANCE - BINARY CLASSIFICATION
###################################
'''
The situation where one class is more frequent is called class imbalance. 
For example if we have a dataset such that 99% of transactions are legitite; 1% are fraudalent,
the class of legitimate transactions contains way more instances than the class of fraudulent transactions.
'''

'''
CONFUSION MATRIX - OVER A BINARY CLASSIFICATION

Given a binary classifier, such as our fraudulent transactions example, 
we can create a 2-by-2 matrix that summarizes performance called a confusion matrix.


Across the top are the predicted labels.
Across the down the side are the actual labels.

Given any model, we can fill in the confusion matrix according to its predictions.

######################
KEYS to interpretate it and do not forget it:
    IS LIKE A DIAGONAL MATRIX, the diagonal entries 
    
    Class 1 with class 1, class 2 with class 2 into the diagonal of the matrix
    
    The diagonal entries contains the number of each class correctly labeled with the model
    
    And the other entries?
    Remember how read a matrix given its size... n*m, the first value corresponds to ROW
    
    So the other entries... read them by its rows
    The other entries tells you the number of incorrectly classes labeled
    The rows indicates you to which class belongs the incorrect label.
######################
    
Usually, the class of interest is called the positive class. 
In our example, the fraudulent label is our interes...
we aim to detect fraud, the positive class is an illegitimate transaction. 

                        
                        BUT WHY IT IS IMPORTANT?
* ACCURACY
    - Firstly, we can retrieve accuracy: it's the sum of true predictions divided by the total sum of the matrix.
    In other words, classic probability, positive cases (from the diagonal entries) / total cases

* PRECISION
    - true positives / (true positives + false POSITIVES)
    Remember
    TN   FP
    FN   TP
    BUT THIS IS NO THE UNIQUE FORM OF CONFUSION MATRIX
    TP   FP
    FN   TN
    At top predicted and DOWN THE SIDE (por el costado) actual classes
    
* Recall or Sensitivelu
    - true positives / (true positives + false NEGATIVES)

* F1 SCORE
    - 2 * ((precision * recall) / (precision + recal))
    
    tp    fp - precision
    fn    tn
    |       \
    recal    accuracy
'''
# Code
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model import KNeighborrsClassifier
model = KNeighborrsClassifier(n_neighbors = 7)

X_train, X_test, y_train, y_test  =train_test_split(X, y, test_size = 0.4, random_state=42)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

confusion_matrix(y_test, y_pred)
# .score works without y_pred, it works with X_test, y_test, testing data

# WE COULD GET THE PRECISION, RECALL AND SUPPORT (WHICH REPRESENTS THE NUMBER OF INSTANCES FOR AEACH CLASS WITHIN THE TRUE LABELS)
print(classification_report(y_test, y_pred))


    