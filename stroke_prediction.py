"""
Dataset: Kaggle Stroke Prediction Dataset

## Load Data
"""

import pandas as pd
import numpy as np
df = pd.read_csv('data/healthcare-dataset-stroke-data.csv')

"""## Preprocessing"""

df.drop(columns=['ever_married'], inplace=True)
df.drop(columns=['work_type'], inplace=True)
df.drop(columns=['Residence_type'], inplace=True)
df.drop(columns=['avg_glucose_level'], inplace=True)

df.isnull().sum()
# 결측치 확인

df['gender'].value_counts()
# gender 데이터별 개수 확인

len(df.loc['smoking_status'] == 'never smoked') & (df['stroke'] == 0)

len(df.loc['smoking_status'] == 'Unknown') & (df['stroke'] == 0)

df.loc[df['smoking_status'] == 'Unknown', 'smoking_status'] = 0
df.loc[df['smoking_status'] == 'never smoked', 'smoking_status'] = 0
df.loc[df['smoking_status'] == 'formerly smoked', 'smoking_status'] = 1
df.loc[df['smoking_status'] == 'smokes', 'smoking_status'] = 2
df = df.astype({'smoking_status':int})
df.info()

df.drop(columns=['id'], inplace=True)
df.drop(df[df['gender'] == 'Other'].index, axis=0, inplace=True)
# id 칼럼, 성별 other 데이터 제거

A = df['bmi'].mean()
df['bmi'] = df['bmi'].fillna(A)
# bmi 결측치 평균값으로 대체

df.info()

df['stroke'].value_counts()
# 발병 여부 데이터 개수 확인

S = df[df['stroke'] == 0]
len(S)

s1 = S.sample(n=250, random_state=1234)
len(s1)

s2 = df[df['stroke'] == 1]
len(s2)

data = pd.concat([s1,s2])
data['stroke'].value_counts()

"""## Correlation Analysis"""

corr = data.corr()
corr

import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(5,4))
sns.heatmap(corr,annot=True, cmap = 'Blues')

# 고혈압 여부와 심장병 여부의 상관관계 (파이계수)
from scipy.stats.contingency import association
X = np.array([[371, 42],
              [68, 18]])
association(X, method="tschuprow")

"""## Logistic Regression"""

X = data[['age', 'hypertension', 'heart_disease']] # 독립변수 (연령, 고혈압 여부, 심장병 여부)
Y = data['stroke'] # 종속변수 (뇌졸중 발병 여부)

from sklearn.model_selection import train_test_split
train_X, test_X, train_Y, test_Y = train_test_split(X, Y)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
train_X = scaler.fit_transform(train_X)
test_X = scaler.transform(test_X)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(train_X, train_Y)

print(model.coef_) # 회귀계수

"""## Evaluation"""

from sklearn.metrics import classification_report

y_pred = model.predict(train_X)
print(classification_report(train_Y, y_pred))

y_pred = model.predict(test_X)
print(classification_report(test_Y, y_pred))

from sklearn.metrics import confusion_matrix
confusion_matrix (test_Y, y_pred)

sns.heatmap(confusion_matrix (test_Y, y_pred),
            annot = True, fmt = "d",cmap = 'Blues')
plt.xlabel('predicted')
plt.ylabel('actual')

print(model.score(train_X, train_Y))

print(model.score(test_X, test_Y))

"""## Examples of Prediction"""

MAN1 = np.array([27, 0, 0])
MAN2 = np.array([45, 1, 0])
MAN3 = np.array([64, 0, 1])
MAN4 = np.array([83, 1, 1])

vsample_df = np.array([MAN1, MAN2, MAN3, MAN4])

sample_df = scaler.transform(sample_df)

print(model.predict(sample_df))
print(model.predict_proba(sample_df))
