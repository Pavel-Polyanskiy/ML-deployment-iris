# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd
import numpy as np 
from sklearn.linear_model import LogisticRegression


# %%
df = pd.read_csv('iris.csv')
df.head()


# %%
df_enc = pd.get_dummies(df)


# %%
mappings = df['variety'].map({'Setosa':0,'Versicolor':1, 'Virginica': 2})
df['label'] = mappings


# %%
X = df_enc.iloc[:, :4]
y = df['label']


# %%
lr = LogisticRegression(max_iter=1000)
lr.fit(X, y)


# %%
from sklearn.metrics import accuracy_score
accuracy_score(y, lr.predict(X))


# %%
lr.predict(X)


# %%
arg_ind = lr.predict_proba(X)[0].argmax()
np.round((lr.predict_proba(X)[0][arg_ind] * 100), 2)


# %%
df.columns


# %%
def predict(sep_len, sep_wid, pet_len, pet_wid):
    to_predict = [sep_len, sep_wid, pet_len, pet_wid]
    to_predict = np.array(to_predict).reshape(1, -1)
    prediction = lr.predict(to_predict)
    if prediction == 0:
        return 'Setosa'
    elif prediction == 1:
        return 'Versicolor'
    else:
        return 'Virginica'
    


# %%
def predict_proba(sep_len, sep_wid, pet_len, pet_wid):
    to_predict = [sep_len, sep_wid, pet_len, pet_wid]
    to_predict = np.array(to_predict).reshape(1, -1)
    arg_ind = lr.predict_proba(to_predict).argmax()
    prediction_proba = str(np.round((lr.predict_proba(to_predict)[0][arg_ind] * 100), 2)) + '%'
    return prediction_proba


# %%
import pickle
pickle.dump(lr, open('/Users/polyanaboss/Desktop/ML Deployment/first_deployment/logreg.pickle', 'wb'))


