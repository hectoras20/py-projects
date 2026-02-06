import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix

from pathlib import Path



# What exactly is credit risk? 
# Credit risk is the risk that someone who has borrowed money will not repay it all.
# A loan is in default when the lending agency is reasonably certain the loan will not be repaid. 
# We will use machine learning models to determine this.

# EXPECTED LOSS
# Expected loss is a simple calculation of the following three components multiplied. 
# 1. The probability of default, which is the likelihood someone will default on a loan. 
# 2. The exposure at default which is the amount outstanding at the time of default. 
# 3. The loss given default which is the ratio of the exposure against any recovery from the loss. 

# TYPES OF DATA 
# For modeling probability of default we generally have two primary types of data available. 
# 1. Application data, which is data that is directly tied to the loan application like loan grade, interes rate, amount. 
# 2. Behavioral data, which describes the recipient(destinatario) of the loan, such as employment length, historical default, income.

# DATA USED
# The data we will use for our predictions of probability of default includes a mix. 
# This is important because application data alone is not as good as application and behavioral data together.
# Some of the columns available in the data set are personal income, the loan amount's percentage of the person's income, and credit history length. 
# Consider the percentage of income. This could affect loan status if the loan amount is more than their income, because they may not be able to afford payments.

# Our data has 32 thousand rows, which can be difficult to see all at once. Here is where we use cross tables using the crosstab function available within Pandas. 
# We can use this function to help get a high level view of the data SIMILAR TO PIVOT TABLES IN EXCEL.

# Your must be in the Credit_Risk_Modeling directory.
try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    BASE_DIR = Path().resolve()

DATA_PATH = BASE_DIR / "data" / "cr_loan.csv"

cr_loan = pd.read_csv(DATA_PATH)
pd.crosstab(cr_loan['person_home_ownership'], cr_loan['loan_status'], values = cr_loan['loan_int_rate'], aggfunc = 'mean').round(2)

# Exploring with visuals
# In addition to using cross tables, we can explore the data set visually. Here, we use matplotlib to create a scatter plot of the loan's interest rate and the recipient's income. 
# Just like the cross table, plots help us get a high level view of our data.

plt.scatter(cr_loan['person_income'], cr_loan['loan_int_rate'], c = 'blue', alpha = 0.5)
plt.xlabel('Personal Income')
plt.ylabel('Loan Interest Rate')
plt.show()

# DATA PREPARATION - OUT FIRST STEP
# When our data is properly prepared we reduce the training time of our machine learning models.
#  Also, prepared data can also have a positive impact on the performance of our model. 
# This is important because we want our models to predict defaults correctly as often as possible.

    # 1. OUTLIER DETECTION AND REMOVAL - Fisrt type of preparation
    # With outliers in our training data, our predictive models will have a difficult time estimating parameters like coefficients. 
    # Think of the coefficients as how much each column or feature is weighted to determine the loan status.
    # Outliers can cause columns to have a much higher or lower weight than normal.
    # Imagine having an interest rate of 59,000 percent! - We could see that with crosstab

# A way to detect outliers is to use visuals.
plt.scatter(cr_loan['person_emp_length'], cr_loan['loan_int_rate'], c = 'blue', alpha = 0.5)
plt.xlabel('Person Employment Lenght')
plt.ylabel('Loan Interest Rate')
plt.show()
# Here, we can see that a couple records have a person's employment length -set at well over(muy superior a)- 100. 
# This would suggest that two loan applicants are over 136 years old! This, for now at least, is not possible.

# So, we know outliers are a problem and want to remove them, but how? 
# We can easily use the drop method within the pandas package to remove rows from our data.
indices = cr_loan[cr_loan['person_emp_length'] >= 60].index
cr_loan.drop(indices, inplace=True)
# Verifier
plt.scatter(cr_loan['person_emp_length'], cr_loan['loan_int_rate'], c = 'blue', alpha = 0.5)
plt.xlabel('Person Employment Lenght')
plt.ylabel('Loan Interest Rate')
plt.show()

    # 2. RISK WITH MISSING DATA
    # So, how do we handle missing data? Most often, it is handled in one of three ways. 
    # Sometimes we need to replace missing values. 
    # * This could be replacing a null with the average value of that column. 
    # * Other times we remove the row with missing data all together. For example, if there are nulls in loan amount, we should drop those rows entirely. 
    # * We sometimes keep missing values as well. This, however, is not the case with most loan data. Understanding the data will direct you towards one of these three actions.
    
    # For example, if the loan status is null, it's possible that the loan was recently processed in our system. 
    # Sometimes there is a data delay, and additional time needed for processing. 
    # In this case, we should just remove the whole row. 
    # Another example is where the person's age is missing. 
    # Here, we might be able to replace the missing age values with the median of everyone's age.

# Finding missing data
null_columns = cr_loan.columns[cr_loan.isnull().any()]
cr_loan[null_columns].isnull().sum()

# Replacing method
# If we decide to replace missing data, we can call the fill-n-a method from Pandas along with aggregate functions. This will replace only missing values
cr_loan['loan_int_rate'].fillna( (cr_loan['loan_int_rate'].mean()) , inplace = True)

# Dropping Method
indices_drop = cr_loan[cr_loan['person_emp_length'].isnull()].index
cr_loan.drop(indices_drop, inplace = True)

cr_loan.dropna()

# Verifier
null_columns = cr_loan.columns[cr_loan.isnull().any()]
cr_loan[null_columns].isnull().sum()


# LOGISTIC REGRESSION FOR POBABILITY OF DEFAULT - SECOND STEP
# Recall that the probability of default is the likelihood that someone will fail to repay a loan. 
# This is expressed as a probability which is a value between zero and one. 
# These probabilities are associated with our LOAN STATUS COLUMN where a 1 is a default, and a 0 is a non default.

# The resulting predictions give us probabilities of default. The closer the value is to 1, the higher the probability of the loan being a default.
# The class is default or non-default in this case...
    
# IN THE INDUSTRY, two models are used frequently. 
# These are logistic regressions, and decision trees. Both of these models can predict the probability of default, and tell us how important each column is for predictions  

    # 1. LOGISCTIC REGRESSION
    # The logistic regression is like a linear regression but only produces a value between 0 and 1. 
    # Notice that the equation for the linear regression is actually part of the logistic regression. 
    # Logistic regressions perform better on data when what determines a default or non-default can vary greatly.

# Using the logistic regression within scikit learn
# Like any function, you can pass in parameters or not. The solver parameter is an optimizer, just like the solver in Excel. LBFGS is the default
clf_logistic_one = LogisticRegression(solver='lbfgs')

# KEY: Interest rates are easy to understand, but what how useful are they for predicting the probability of default?
X_one = cr_loan[['loan_int_rate']]
y = cr_loan[['loan_status']]

clf_logistic_one.fit(X_one, np.ravel(y))

# Printing the parameters of the model
print('COEFFICIENTS OF THE MODEL with a single feature: ', clf_logistic_one.coef_)

# Printing the intercept of the model
print('INTERCEPT OF THE MODEL with a single feature: ', clf_logistic_one.intercept_)

# Generally, we won't use only loan_int_rate to predict the probability of default. We will want to use all the data you have to make predictions.
# Will this model differ from the first one? 
# For this, we can easily check the .intercept_ of the logistic regression. 
# REMEMBER that this is the y-intercept of the function and the overall log-odds of non-default.

X_multi = cr_loan[['loan_int_rate','person_emp_length']]

# Creating and training a new logistic regression
clf_logistic_multi = LogisticRegression(solver='lbfgs').fit(X_multi, np.ravel(y))

# Print the intercept of the model
print('INTERCEPT OF THE MODEL with two features:', clf_logistic_multi.intercept_)

# NOW, the new clf_logistic_multi model has an .intercept_ value closer to zero. This means the log odds of a non-default is approaching zero.

# Our workflow is now:
X = cr_loan[['loan_int_rate','person_emp_length','person_income']]

# Generally, in machine learning, we split our entire data set into two individual data sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4, random_state=123)

clf_logistic = LogisticRegression(solver='lbfgs').fit(X_train, np.ravel(y_train))

print('COEFFICIENTS OF THE MODEL with 3 features :', clf_logistic.coef_)

########################################################################################################################
# THE BEST PART - UNDESTANDING THE MODEL
# PREDICTING THE PROBABILITY OF DEFAULT - HOW IT WORKS?
# We previously saw the intercept and coefficients for our model. 
# These coefficients the importance of each column.

# "Each coefficient is multiplied by the values in the column, and then added together along with the intercept. Then, 1 is divided by the sum of 1 and e to the negative power of our intercept coefficient sums. The result is the probability of default". 
intercept = clf_logistic.intercept_[0]
coefficients = clf_logistic.coef_[0]

coef_int_rate = coefficients[0]
coef_emp_length = coefficients[1]
coef_income = coefficients[2]

print(intercept, coef_int_rate, coef_emp_length, coef_income)

# AND NOW, THIS IS WHAT THE MODEL DOES:
# LET US TAKE AN OBSERVATION FROM THE TEST SET
sample = X_test.iloc[0]

loan_int_rate = sample['loan_int_rate']
person_emp_length = sample['person_emp_length']
person_income = sample['person_income']

# 1. Linear sum - log-odds
linear_sum = (
    intercept
    + coef_int_rate * loan_int_rate
    + coef_emp_length * person_emp_length
    + coef_income * person_income
)

# 2: Logistic Function (sigmoid)
prob_default = 1 / (1 + np.exp(-linear_sum))
prob_non_default = 1 - prob_default

print("Default Probability:", prob_default)
print("No default Probability:", prob_non_default)

# Validator
model_prob = clf_logistic.predict_proba(X_test.iloc[[0]])[0][1]

print("Manual Probability:", prob_default)
print("Sklearn Probability:", model_prob)
# BOOOM

########################################################################################################################
# INTERPRETING COEFFICIENTS - ANOTHER KEY 
# Consider employment length as an example. I've already calculated the intercept and coefficient for a logistic regression using this one column.
X_one = cr_loan[['person_emp_length']]

clf_logistic_one.fit(X_one, np.ravel(y))
print('COEFFICIENTS OF THE MODEL with a single feature: ', clf_logistic_one.coef_)
print('INTERCEPT OF THE MODEL with a single feature: ', clf_logistic_one.intercept_)

# What this coefficient tells us is the log odds for non-default. 
# This means that for every 1 year increase in employment length, the person is less likely to default by a factor of the coefficient.
person_emp_length_sample = np.arange(1, 21).reshape(-1, 1)
probability_of_default = clf_logistic_one.predict_proba(person_emp_length_sample)[:, 1] # We take all the row values BUT those that are in the second column, since these are the default probabilities that concerned us. 
plt.figure()
plt.plot(person_emp_length_sample, probability_of_default)
plt.xlabel("Employment Length (years)")
plt.ylabel("Probability of Default")
plt.title("Effect of Employment Length on Probability of Default")
plt.show()
# What we see here is that the higher a person's employment length is, the less likely they are to default.

# We can use loan_int_rate to see that the higher the interest rates, the GREATER the probability of deafult we could have!!!!!
X_one = cr_loan[['loan_int_rate']]

clf_logistic_one.fit(X_one, np.ravel(y))
print('COEFFICIENTS OF THE MODEL with one single feature: ', clf_logistic_one.coef_)
print('INTERCEPT OF THE MODEL with one single feature: ', clf_logistic_one.intercept_)

# What this coefficient tells us is the log odds for non-default. 
# This means that for every 1 year increase in employment length, the person is less likely to default by a factor of the coefficient.
person_emp_length_sample = np.arange(1, 21).reshape(-1, 1)
# Probabilities = [[non-deafult, default]]
probability_of_default = clf_logistic_one.predict_proba(person_emp_length_sample)[:, 1] # We take all the row values BUT those that are in the second column, since these are the default probabilities that concerned us. 
plt.figure()
plt.plot(person_emp_length_sample, probability_of_default)
plt.xlabel("Loan Interest Rate")
plt.ylabel("Probability of Default")
plt.title("Effect of Loan Interest Rate on Probability of Default")
plt.show()

# USING NO NUMERIC COLUMNS... ONE HOT ENCODING
    # The main idea is to represent a string with a numeric value.
    # For this, we use the get dummies function within pandas.
    # 1. First, we separate the numeric and non-numeric columns from the data into two sets.
    # 2. Then we use the get dummies function to one-hot encode only the non-numeric columns. 
    # 3. We union the two sets and the result is a full data set that's ready for machine learning!

cred_num = cr_loan.select_dtypes(exclude=['object'])

cred_cat = cr_loan.select_dtypes(include=['object'])

cred_cat_onehot = pd.get_dummies(cred_cat)

cr_loan = pd.concat([cred_num, cred_cat_onehot], axis=1)

# Printing the columns in the new data set
print(cr_loan.columns)

# Now we can do the same workflow to set a model...
X = cr_loan.drop('loan_status', axis=1).dropna()
y = cr_loan[['loan_status']]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=123) 

clf_logistic = LogisticRegression(solver='lbfgs').fit(X_train, np.ravel(y_train))

preds = clf_logistic.predict_proba(X_test)

# Create dataframes of first five predictions, and first five true labels
preds_df = pd.DataFrame(preds[:,1][0:5], columns = ['prob_default'])
true_df = y_test.head(5)

# Concatenate and print the two data frames for comparison
print(pd.concat([true_df.reset_index(drop = True), preds_df], axis = 1))


########################################################################################################################
# CREDIT MODEL PERFORMANCE
# The easiest way to analyze performance is with accuracy.
# One way to check this is to use the score method within scikit-learn on the logistic regression. 
# This is used on the trained model and returns the average accuracy for the test set. 
# Using the score method will display this accuracy as a percentage.

clf_logistic.score(X_test, y_test)

# ROC CURVE CHARTS
# R-O-C charts are a great way to visualize the performance of our model.
# They plot the true positive rate, the percentage of correctly predicted defaults, against the false positive rate, the percentage of incorrectly predicted defaults.

prob_default = preds[:, 1] #KEY
fallout, sensitivity, threshold = roc_curve(y_test, prob_default)

plt.figure()
plt.plot(fallout, sensitivity, color = 'darkorange')
plt.plot(fallout, fallout, color = 'blue', linestyle='--')
plt.show()
# The  blue line represents a random prediction
# The orange line represents our model's predictions.

# But how can we interpretate it?
    # R-O-C charts are interpreted by looking at how far away the model's curve (ORANGE LINE) gets from the dotted blue line shown here, which represents the random prediction. 
    # This movement away from the line is called lift. 
    # The more lift we have, the larger the area under the curve gets, which is something that we want.
    # The A-U-C is the calculated area between the curve and the random prediction. 
    # This is a direct indicator of how well our model makes predictions.
auc = roc_auc_score(y_test, prob_default)  # actual defaults vs prob. computed

# THRESHOLDS

# 1. We will first need to create a variable to store the predicted probabilities. We already did this point with the variable preds = clf_logistic.predict_proba(X_test)
# 2. Then we can create a data frame from the second column OF PREDS which contains the probabilities of default. Remember preds = [prob. non deault, prob. default]
# 3. Then we apply a quick function to assign a value of 1 if the probability of default is above our threshold of 0.5. 

preds = clf_logistic.predict_proba(X_test) # KEY
preds_df = pd.DataFrame(preds[:, 1], columns = ['prob_default']) 
preds_df['loan_status'] = preds_df['prob_default'].apply(lambda x : 1 if x > 0.5 else 0)

print(preds_df['loan_status'].value_counts())
# The result of this is a data frame with new values for loan status based on our threshold.

# THRESHOLD WORKS WITH THE CONFUSSION MATRIX, so we can se if we get a better performance setting new thresholds. 
#####################################
# CREDIT CLASSIFICATION REPORTS - NEW
# This will show us several different evaluation metrics all at once! 
# We use this function to evaluate our model using our true values for loan status stored in the y_test set, and our predicted loan status values from our logistic regression and the threshold we set.

target_names = ['Non-Default', 'Default']
class_report = classification_report(y_test, preds_df['loan_status'], target_names = target_names)

# There are 2 really useful metrics in this table, and they are the precision and recall. 
# But for now, let's focus on recall.
# RECALL
    # The definition of default recall, also called sensitivity, is the proportion of actual positives correctly predicted.
    # Recall - Default = That means we correctly predicted ___ percent of defaults, and incorrectly predicted __ percent of defaults (complement of the first given value)
    
    # Precision recall fscore support function within sci-kit learn. 
    # With this function, we can get the recall for defaults from by subsetting the report the way we would any array. Here we select the second value from the second set.

precision_recall_fscore_support(y_test, preds_df['loan_status'])[1][1]
#################################
# MODEL DISCRIMINATION AND IMPACT
# Confusion Matrix

# Since we already did: preds_df['loan_status'] = preds_df['prob_default'].apply(lambda x : 1 if x > 0.5 else 0)
# With the previus line of code we can set new thresholds.
print(confusion_matrix(y_test,preds_df['loan_status']))
# We can adjust the threshold and see if we get a better performance-

# Precision(0) = TN/(TN+FN) ||
# Precision(1) = TP/(TP+FP) ||
# Recall(0) = TN/(TN+FP) -- 
# Recall(1) = TP/(TP+FN) --

# IDEA
results = pd.DataFrame(columns=['threshold', 'non-def_recall', 'default_recall', 'estimated impact loss'])
avg_loan_amnt = 50 # dollas
row = 0  

for threshold in np.arange(0, 1.1, 0.1):
    preds_df['loan_status'] = preds_df['prob_default'].apply(lambda x: 1 if x > threshold else 0)

    conf_matrix = confusion_matrix(y_test, preds_df['loan_status'])

    # Recall = TP / (TP + FN)
    default_recall = conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[1, 0])
    non_def_recall = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1])

    num_defaults = preds_df['loan_status'].value_counts().get(1, 0)

    # Calculating the estimated impact of the new default recall rate
    impact = avg_loan_amnt * num_defaults * (1 - default_recall)

    results.loc[row] = [threshold, non_def_recall, default_recall, impact]
    row += 1  
# That means we correctly predicted ___ percent of defaults, and incorrectly predicted __ percent of defaults (complement of the first given value)

# Ploting performance
plt.plot(results['threshold'], results['default_recall'])
plt.plot(results['threshold'], results['non-def_recall'])
plt.xlabel('Probability Threshold')
plt.legend(['Default Recall", "Non-default Recall'])
plt.show() 

# INTERPRETATION
# Approximately what starting threshold value would maximize these scores evenly? Looking at the graph... 0.275
# Because it's the point where all lines converge. 
# This threshold would make a great starting point, but declaring all loans about 0.275 to be a default is probably not practical.

########################################################################################################################
# GRADIENT BOOSTED TREES WITH XGBOOST
# Decision Trees are machine learning models which use decisions as steps in a process to eventually identify our loan status.
# The xgboost package train similar to logistic regression models.

clf_gbt = xgb.XGBClassifier()
clf_gbt.fit(X_train, np.ravel(y_train))

# And we still do the same workflow...
# Predicting with a model... We can use predict_proba to predict probabilities of default. 
gbt_preds = clf_gbt.predict_proba(X_test)

preds_df = pd.DataFrame(gbt_preds[:,1][0:5], columns = ['prob_default'])
true_df = y_test.head(5)

# Concatenate and print the two data frames for comparison
print(pd.concat([true_df.reset_index(drop = True), preds_df], axis = 1))

# AND USING PREDICT INSTEAD OF PREDICT_PROBAS, we can do things like...
gbt_preds = clf_gbt.predict(X_test)

target_names = ['Non-Default', 'Default']
print(classification_report(y_test, gbt_preds, target_names=target_names))

# These also have hyperparameters that affect how the model learns
# HYPERPARAMETERS CANNOT BE LEARNED FROM DATA, THEY HAVE TO BE SET BY US
    # 1. The learning rate tells the model how quickly it should learn in each step of the ensemble. 
    # The smaller the value, the more conservative it is at each step. 
    # 2. The max depth tells the model how deep each tree can go. 
    # Keeping this value low ensures the model is not too complex.

xgb.XGBClassifier(learningn_rate = 0.2,
                  max_depth = 4)



