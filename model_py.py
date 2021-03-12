import warnings

warnings.filterwarnings('ignore')


# In[2]:


# Importing the required libraries

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from tpot import TPOTClassifier
from sklearn.metrics import confusion_matrix,accuracy_score,roc_auc_score

import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


filepath = "transfusion.data"

# above .data file is comma delimited

df = pd.read_csv(filepath, delimiter=",")
df.head()

df.info()


# In[5]:


print("Rows : ", df.shape[0],"\n")
print("Columns : ", df.shape[1])


# In[6]:


df.describe()

df.rename(columns={'whether he/she donated blood in March 2007': 'target'},inplace=True)

df.head()

X_Data = df.drop(columns='target')
X_Data.head(2)


# In[33]:


Y_Data = df.target
Y_Data.head(2)


X_train, X_test, y_train, y_test = train_test_split(X_Data,Y_Data,test_size=0.3,random_state=0,stratify=df.target)


# In[35]:


X_train.head()

tpot = TPOTClassifier(
    generations=5, #number of iterations to run ; pipeline optimisation process ; by default value is 100
    population_size=20, #number of individuals to retrain in the genetic programing popluation in every generation, by default value is 100
    verbosity=2, #it will state how much info TPOT will communicate while it is running
    scoring='roc_auc', #use to evaluate the quality of given pipeline
    random_state=42,
    disable_update_check=True,
    config_dict='TPOT light'
)
tpot.fit(X_train, y_train)

tpot_auc_score = roc_auc_score(y_test, tpot.predict_proba(X_test)[:, 1])
print(f'\nAUC score: {tpot_auc_score:.4f}')

# Print best pipeline steps
print('\nBest pipeline steps:', end='\n')
for idx, (name, transform) in enumerate(tpot.fitted_pipeline_.steps, start=1):
    # Print idx and transform
    print(f'{idx}. {transform}')


tpot.fitted_pipeline_

col_norm = ["Monetary (c.c. blood)"]

# Copy X_train and X_test into X_train_normed and X_test_normed
X_train_normed, X_test_normed = X_train.copy(), X_test.copy()

# Log normalization
for df_norm in [X_train_normed, X_test_normed]:
    # Add log normalized column
    df_norm['log_monetary'] = np.log(df_norm[col_norm])
    # Drop the original column
    df_norm.drop(columns=col_norm, inplace=True)


print("X_train Value\n")
print(X_train.head())
print("------------------------")
print("X_train_normed Value\n")
print(X_train_normed.head())


print("X_train Variance\n")
print(X_train.var().round(2))
print("------------------------")
print("X_train_normed Variance\n")
print(X_train_normed.var().round(2))




from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(C=25.0, random_state=42)
# Train the model
logreg.fit(X_train_normed, y_train)


#predicting on the test data
prediction = logreg.predict(X_test)

#Confusion matrix
confusion_matrix(prediction,y_test)


logreg_auc_score = roc_auc_score(y_test, logreg.predict_proba(X_test_normed)[:, 1])
print(f'\nAUC score: {logreg_auc_score:.4f}')


from operator import itemgetter

# Sort models based on their AUC score from highest to lowest
sorted(
    [('tpot', tpot_auc_score), ('logreg', logreg_auc_score)],
    key=itemgetter(1),
    reverse=True)

import pickle

# Saving model to disk
pickle.dump(logreg, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl', 'rb'))
print(model.predict([[9, 3, 750, 52]]))