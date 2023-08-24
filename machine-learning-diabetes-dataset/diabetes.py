
### Diabets Dataset ###

import pandas as pd
from pandas.core.reshape.concat import concat
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,roc_curve


diabetes = pd.read_csv("diabetes.csv")
diabetes_df = diabetes.copy()
diabetes_df["Outcome"] = diabetes_df["Outcome"].astype("category")

print("********** Data **********")
print(diabetes_df)
print("********** Null Control **********")
print(diabetes_df.isna().sum())
print("********** İnfo **********")
print(diabetes_df.info())
print("********** Describe **********")
print(diabetes_df.describe().T)
print("********** Features **********")
print(diabetes_df.columns)
print("********** Labels **********")
print(diabetes_df.columns[8])

### Train ###

x = diabetes_df.drop(["Outcome"],axis=1)
y = diabetes_df["Outcome"]

X_train,X_test,Y_train,Y_test = train_test_split(x,y, test_size=0.2)

model = LogisticRegression(max_iter = 200).fit(X_train,Y_train)

print("********** X Train Data **********")
print(X_train)
print("********** X Test Data **********")
print(X_test)
print("********** Y Train Label **********")
print(Y_train)
print("********** Y Test Label **********")
print(Y_test)

print("********** Model **********")
print(f"Y = X1*{model.coef_[0][0]} + X2*{model.coef_[0][1]} + X3*{model.coef_[0][2]} + X4*{model.coef_[0][3]} + X5*{model.coef_[0][4]} + X6*{model.coef_[0][5]} + X7*{model.coef_[0][6]} + X8*{model.coef_[0][7]} + {model.intercept_[0]}")
print("Score = ",model.score(X_train,Y_train))

### Prediction ###

prediction_diabetes_df = pd.DataFrame(model.predict(x),columns=["Perdiction"])
diabetes_df = concat([diabetes,prediction_diabetes_df],axis=1)
print("********** Prediction Data **********")
print(diabetes_df)

### Performance ###

prediction = model.predict(X_test)
confusion_diabetes_df = pd.DataFrame(confusion_matrix(Y_test,prediction),columns=["Predicted Positive","Predicted Negative"],index=["Actual Positive","Actual Negative"])
print("********** Confusion Matris **********")
print(confusion_diabetes_df)
print("********** Confusion Matris Metrics **********")
print("Accuary = ",accuracy_score(Y_test,prediction))
print("Precision = ",precision_score(Y_test,prediction))
print("Recall = ",recall_score(Y_test,prediction))
print("F1 Score = ",f1_score(Y_test,prediction))

print("********** Roc & Auc **********")
print("Roc & Auc Score = ",roc_auc_score(Y_test,model.predict(X_test)))

model_roc_auc = roc_auc_score(Y_test, model.predict(X_test))

fpr, tpr, thresholds = roc_curve(Y_test, model.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='AUC (area = %0.2f)' % model_roc_auc)
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim(([0.0, 1.0]))
plt.ylim(([0.0, 1.05]))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC')
plt.show()

