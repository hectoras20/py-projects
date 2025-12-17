'''
If we have a problem where we know the class which the element belongs (it is a column with the real values)
we are talking about a supervised learning model

There are two types of supervised learning. 
- Classification is used to predict the label, or category, of an observation. For example, we can predict whether a bank transaction is fraudulent or not. As there are two outcomes here - a fraudulent transaction, or non-fraudulent transaction, this is known as binary classification. 
- Regression is used to predict continuous values. For example, a model can use features such as number of bedrooms, and the size of a property, to predict the target variable, price of the property.
'''

"""
STEP 1.
There are some requirements to satisfy before performing supervised learning…
- Our data must not have missing values
- Must be in numeric format
- Stored as pandas DataFrames or Series, or NumPy arrays. 

This requires some exploratory data analysis first to ensure data is in the correct format. 
Various pandas methods for descriptive statistics, along with appropriate data visualizations, are useful in this step.
"""

# Let's create the code, this will allow us to create the class more easily later!

import pandas as pd
import numpy as np
df_name = 'iris_con_genero'
df = pd.read_csv('/Users/hectorastudillo/py-proyects/machine_learning/data/' + df_name + '.csv')
df.info()


"""
STEP 2. 
Scikit-learn follows the same syntax for all supervised learning models, 
which makes the workflow repeatable. 
Let's familiarize ourselves with the general scikit-learn workflow syntax. 

from sklearn.module import Model
modeñ = Model() - initialization
model.fit(X,y)
predictions = model.predict(X_new)

* KNN
Is to predict the label of any data point by looking at the k, for example, three, closest labeled data points and getting them to vote on what label the unlabeled observation should have. 
KNN uses majority voting, which makes predictions based on what label the majority of nearest neighbors have. 

* Logistic Regression = Binary Classification 
"""
# K-Nearest Neighbors - Supervised Learning Model to Classification
# df.plot_original_data()
# df.apply_model(type = str, arguments that need the model = None) - KNN and K

# Code
from sklearn.neighbors import KNeighborsClassifier
target_column = 'Species' # Name of the column to sort
# In the following list we could add the columns that does not meet the requirements or columns to skip, we could filter such columns (as a idea).
skip_columns = []

# Obtaining data
X_columns = [i for i in df.columns if i != target_column and i not in skip_columns]

X = df[X_columns].values
y = df[target_column].values

"""
OBSERVATION about the size of X and y in both types of supervised learning.
To regression there is not problem with the size of the data that we want to predict, these makes an extrapolation (y^ values)
There is also not problem with KNN with Scikit when we want to make predictions (y^)
In both, when we train our model… X and y must have the same size! To fit our model
"""
print(X.shape, y.shape)
# We will obtain that the shape of X (features) is 145 values and 4 columns (145, 4), while the shape of y (target) is 145 values and 1 column (145, )


# Ploting original data



knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(X, y)
# Now we need to pass data, we could have a dataframe that only contains the futures, and use it to predict...
name_df_predict = 'iris_sin_genero'
X_new = pd.read_csv('/Users/hectorastudillo/py-proyects/machine_learning/data/' + name_df_predict + '.csv')
predictions = knn.predict(X_new)
print('predictions:{}'.format(predictions))

# Ploting original data with predictions


"""
STEP 3
Evaluating Model Performance
- Accuaracy
Accuracy is based on classical probability… correct predictions / total observations
We can compute it on the data used to fit the classifier… iris_con_genero

However, as this data is used to train the model, performance will not be indicative 
of how well it can generalize to unseen data, which is what we are interested in, 
is like try to make a measure of how well are the predictions using the data predicted.

That is why we split the original data (iris_con_genero) in parts for training set and test set!
We use the test data to calculate accuracy.

We call train_test_split, passing:
- Our features and targets. 
- By setting the test_size argument to zero-point-three we use 30% here.
- Commonly use 20-30% of our data as the test set. 
- The random_state argument sets a seed for a random number generator that splits the data. Using the same number when repeating this step allows us to reproduce the exact split and our downstream results. 
- !!!!! It is best practice to ensure our split reflects the proportion of labels in our data. 
So if churn occurs in 10% of observations, we want 10% of labels in our training and test sets to represent churn. We achieve this by setting stratify equal to y.

In this case we will ONLY work with the complete data (iris_CON_genero) SPLITING it.
"""
# Code
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 21, stratify = y)

# Train data is to FIT/train classifier 
# Test data to PREDICT and calculate the acuracy - we will predict with this and since is the original data, we have the cprrect outcomes to compare it!

knn.fit(X_train, y_train)

# Obtaining the accuracy with the score method
print(knn.score(X_test, y_test))
# With this we could obtain what n_neighbors the models works better

'''
...
Score method do the following:
    KEY: THE .SCORE FUNCTION WILL NOT HAVE THE SAME FORMULA BETWEEN ALL THE MODELS!

IN CLASSIFICATION PROBLEMS
y_pred = knn.predict(X_test)
accuracy = np.mean(y_pred == y_test)
return accuracy

Predice las clases de X_test.
Compara cada predicción con la clase verdadera (y_test).
Calcula el porcentaje de aciertos (accuracy).
Entonces, predice y luego compara con las etiquetas verdaderas.

IN REGRESSION PROBLEMS
y_pred = knn.predict(X_test)
R^2 = 1 - ((y_test - y_pred) ** 2).sum() / ((y_test - y_test.mean()) ** 2).sum()
return R^2
Predice los valores numéricos de X_test.
Calcula el coeficiente de determinación R².

R² mide qué tan bien se ajustan las predicciones a los valores reales.
Un valor de 1 significa predicción perfecta, 0 significa "tan malo como predecir la media", y negativo significa "peor que eso".
...
'''

"""
STEP 4
Model Complexity
Let's discuss how to interpret k. 
Key: The value of k could lead us to under or overfitting models.

Simpler models are less able to detect relationships in the dataset, which is known as underfitting. 
In contrast, complex models can be sensitive to noise in the training data, rather than reflecting general trends. This is known as overfitting.
"""
train_accuracies = {}
test_accuracies = {}

for i in np.arange(1, 26):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    train_accuracies[i] = knn.score(X_train, y_train) # We predict with the training data and we make the comparation between the y_predicted with y_train
    test_accuracies[i] = knn.score(X_test, y_test) # We do the same but with the testing data, we previusly did this in the STEP 3
    print(f"k={i:2d} | train={train_accuracies[i]:.3f} | test={test_accuracies[i]:.3f}")
    
# Ploting the outcomes
import matplotlib.pyplot as plt
plt.figure(figsize = (8,6))
plt.title('KNN: Varying Number of Neighbords')
neighbords = np.arange(1, 26)
plt.plot(neighbords, train_accuracies.values(), label = 'Training Accuracy')
plt.plot(neighbords, test_accuracies.values(), label = 'Testing Accuracy')
plt.legend()
plt.xlabel('Number Of Neighbors')
plt.ylabel('Accuracy')
plt.show()





