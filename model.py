
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
import pickle
# %matplotlib inline


data=pd.read_csv('https://github.com/silpasath/churn_dataset/raw/master/CustomerChurn.csv')

data['TotalCharges']=pd.to_numeric(data['TotalCharges'], errors='coerce')

meanTotalCharge = data.TotalCharges.mean()
data['TotalCharges']=data['TotalCharges'].fillna(meanTotalCharge)

data['Churn'] =data['Churn'].map({'Yes':1,'No':0})

data = data[['tenure','MonthlyCharges','TotalCharges','Churn']]
# Model building
X = data.drop(['Churn'], axis=1)
y = data.Churn
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=100, stratify=y)

xgb = XGBClassifier(n_estimators=800, n_jobs=-1)
xgb.fit(X_train,y_train)
y_pred = xgb.predict(X_test)

predictions = [round(value) for value in y_pred]

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

# Saving model to disk
pickle.dump(xgb, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict(X_test))

