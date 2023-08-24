
### Wine Dataset ###

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import datasets
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score

wine = datasets.load_wine()
wine_df = pd.DataFrame(wine.data,columns=wine.feature_names)
wine_df['target'] = wine.target


print("********** Data **********")
print(wine_df)
print("********** Null Control **********")
print(wine_df.isna().sum())
print("********** İnfo **********")
print(wine_df.info())
print("********** Features **********")
print(wine_df.columns)
print("********** Labels **********")
print(wine_df.columns[13])

sns.lmplot(x="color_intensity",y="alcohol",fit_reg = False,hue="target",data = wine_df)
plt.show()

## Train ##

x = wine_df.drop(["target"],axis=1)
y = wine_df["target"].astype("category")

X_train,X_test,Y_train,Y_test = train_test_split(x,y,test_size=0.3)

print("********** X Train Data **********")
print(X_train)
print("********** X Test Data **********")
print(X_test)
print("********** Y Train Label **********")
print(Y_train)
print("********** Y Test Label **********")
print(Y_test)

logistic_model = LogisticRegression().fit(X_train,Y_train)
predicion = logistic_model.predict(X_test)
print(accuracy_score(Y_test,predicion))


knn_model = KNeighborsClassifier(n_neighbors=5).fit(X_train,Y_train)
print(accuracy_score(Y_test,knn_model.predict(X_test)))

knn_tunning = KNeighborsClassifier()
param_grid = {"n_neighbors":np.arange(1,25)}
knn_gs = GridSearchCV(knn_tunning,param_grid,cv=5).fit(X_train,Y_train)
print("Bets Parmeter = ",knn_gs.best_params_)
print("Accuracy = ",knn_gs.best_score_)