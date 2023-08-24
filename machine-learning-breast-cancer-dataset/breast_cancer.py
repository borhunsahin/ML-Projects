
### Breast Cancer Dataset ###

from pandas.core.reshape.concat import concat
from sklearn.datasets import load_breast_cancer

import pandas as pd
import numpy as np
from scipy import stats

import matplotlib.pyplot as plt
import seaborn as sns

breast_cancer = load_breast_cancer()
feature_df = pd.DataFrame(data=breast_cancer.data,columns=breast_cancer.feature_names)
label_df = pd.DataFrame(breast_cancer.target,columns=["cancer"])

cancer_df = pd.DataFrame(concat([feature_df,label_df],axis=1))

print("********** Data **********")
print(cancer_df)
print("********** Null Control **********")
print(feature_df.isna().sum())
print("********** İnfo **********")
print(feature_df.info())
print("********** Describe **********")
print(feature_df.describe().T)
print("********** Features **********")
print(breast_cancer.feature_names)
print("********** Labels **********")
print(label_df.columns[0])

### Visualition ###

sns.lmplot(x="mean radius",y="mean area",fit_reg = False,hue="cancer",data = cancer_df,legend=True)
sns.displot(cancer_df["mean area"])
plt.show()

# Outlier Detection

z = np.abs(stats.zscore(feature_df))

outliers = list(set(np.where(z>3)[0]))

print(len(outliers))

new_df = feature_df.drop(outliers,axis = 0).reset_index(drop=False)

y_new = label_df[list(new_df["index"])]











