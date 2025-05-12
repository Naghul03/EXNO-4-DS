# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
```
import pandas as pd
import numpy as np
df=pd.read_csv('/content/bmi.csv')
df
```
![image](https://github.com/user-attachments/assets/1aba1d4f-d782-4909-8c21-ccb393577e10)
```
df.head()
```
![image](https://github.com/user-attachments/assets/7cb51d37-ed4d-4248-a84f-4f904e40d8c5)
```
df.dropna()
```
![image](https://github.com/user-attachments/assets/bf0af2c5-8242-47e9-80cf-df54034ffdb6)
```
max_vals=np.max(np.abs(df[['Height','Weight']]))
max_vals
```
![image](https://github.com/user-attachments/assets/aba4b9b0-2acb-4e50-a9ea-abce9d1397df)
```
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df.head(10)
```
![image](https://github.com/user-attachments/assets/0873cea2-686e-4b12-8e3a-a3c7758d8dcc)
```
df1=pd.read_csv("/content/bmi.csv")
```
```
df2=pd.read_csv("/content/bmi.csv")
```
```
df3=pd.read_csv("/content/bmi.csv")
```
```
df4=pd.read_csv("/content/bmi.csv")
```
```
df5=pd.read_csv("/content/bmi.csv")
```
```
df1
```
![image](https://github.com/user-attachments/assets/fdd20c39-edd3-423e-b17c-f3e6287679c1)
```
from sklearn.preprocessing import StandardScaler
sd=StandardScaler()
df1[['Height','Weight']]=sd.fit_transform(df1[['Height','Weight']])
df1.head(10)
```
![image](https://github.com/user-attachments/assets/ad712247-4bcc-48ac-91a5-654f3133944a)
```
from sklearn.preprocessing import Normalizer
scaler=Normalizer()
df2[['Height','Weight']]=scaler.fit_transform(df2[['Height','Weight']])
df2
```
![image](https://github.com/user-attachments/assets/692e452d-00cf-419f-8dea-582142bdbb8f)
```
from sklearn.preprocessing import MaxAbsScaler
scaler=MaxAbsScaler()
df3[['Height','Weight']]=scaler.fit_transform(df3[['Height','Weight']])
df3
```
![image](https://github.com/user-attachments/assets/42468d80-bce7-45c2-88f4-3fc6b20edbc3)
```
from sklearn.preprocessing import RobustScaler
scaler=RobustScaler()
df4[['Height','Weight']]=scaler.fit_transform(df4[['Height','Weight']])
df4
```
![image](https://github.com/user-attachments/assets/2af75265-9efa-421d-8edc-49b673841b4c)
```
import seaborn as sns
import pandas as pd
from sklearn.feature_selection import SelectKBest,f_regression,mutual_info_regression
from sklearn.feature_selection import chi2
data=pd.read_csv("/content/bmi.csv")
data
```
![image](https://github.com/user-attachments/assets/59fc1e66-e3a1-40d1-ab82-ac6251a512eb)
```
data=pd.read_csv("/content/titanic_dataset (2).csv")
data
```
![image](https://github.com/user-attachments/assets/8c8eb30a-83f6-43fa-9977-955bf9898968)

```
data["Sex"]=data["Sex"].astype("category")
data["Cabin"]=data["Cabin"].astype("category")
data["Embarked"]=data["Embarked"].astype("category")
data
```
![image](https://github.com/user-attachments/assets/f7f56941-11b8-454e-b931-04bd8052c68a)
```
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
y = data["Survived"]
x = data.drop(columns=["Survived", "Name", "Ticket"])
x = x.fillna({
    col: x[col].mean() if x[col].dtype != 'object' else "Missing"
    for col in x.columns
})
x_encoded = pd.get_dummies(x)
selector = SelectKBest(score_func=f_classif, k=5)
x_new = selector.fit_transform(x_encoded, y)
selected_feature_indices = selector.get_support(indices=True)
selected_features = x_encoded.columns[selected_feature_indices]
print("Selected Features:")
print(selected_features)
```
![image](https://github.com/user-attachments/assets/19c0ef9f-46db-44a5-a74d-8f0361a76247)
```
selector = SelectKBest(score_func=f_classif, k=5)
x_new = selector.fit_transform(x_encoded, y)
selected_feature_indices = selector.get_support(indices=True)
selected_features = x_encoded.columns[selected_feature_indices]
print("Selected features")
print(selected_features)

```
![image](https://github.com/user-attachments/assets/b40f96d0-019a-4f0e-aef5-96b708bb619d)

```
x['Cabin'] = x['Cabin'].fillna('Missing')
for col in x.select_dtypes(include=['float64', 'int64']).columns:
    x[col] = x[col].fillna(x[col].mean())
x = pd.get_dummies(x)
selector = SelectKBest(score_func=mutual_info_classif, k=5)
x_new = selector.fit_transform(x, y)
selected_kbest = x.columns[selector.get_support()]
print("Selected Features:")
print(selected_kbest.tolist())
```
![image](https://github.com/user-attachments/assets/a1b119d0-a466-46b6-ad97-86f5d6eb123e)
```
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
sfm = SelectFromModel(model, threshold='mean')
sfm.fit(x, y)
selected_sfm = x.columns[sfm.get_support()]
print("Selected Features:")
print(selected_sfm.tolist())
```
![image](https://github.com/user-attachments/assets/30f48348-dbd6-464b-8964-cda0c8326147)
```
model.fit(x, y)
importances = model.feature_importances_
threshold = 0.1
selected_rf = x.columns[importances > threshold]
print("Selected Features:")
print(selected_rf.tolist())
```
![image](https://github.com/user-attachments/assets/5a6511a0-c8ad-4625-a3f7-ffc39e65800d)
```
model.fit(x, y)
importances = model.feature_importances_
threshold = 0.15
selected_rf = x.columns[importances > threshold]
print("Selected Features:")
print(selected_rf.tolist())
```
![image](https://github.com/user-attachments/assets/370abad9-52f7-4704-920c-345d26ba4307)
```
model=RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(x,y)
feature_importances=model.feature_importances_
threshold=0.15
selected_features = x.columns[feature_importances>threshold]
print("Selected Features:")
print(selected_features)
```
![image](https://github.com/user-attachments/assets/5ca7f347-2278-4d15-b48f-ca568cf8ed89)

# RESULT:
Thus, Feature Scaling and Feature Selection has been used on the given data set.

