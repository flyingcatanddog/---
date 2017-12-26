# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 10:27:15 2017

@author: zhangzerong
"""

#xgboost
#数据集是kaggle比赛中的泰坦尼克号数据集
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import numpy as np

train_df = pd.read_csv('train.csv', header=0)
test_df = pd.read_csv('test.csv', header=0)

#填充值：如果是数值型，则填充均值；如果是字符型，填充最多的类型
from sklearn.base import TransformerMixin
class DataFrameImputer(TransformerMixin):
    def fit(self, X, y=None):
        self.fill = pd.Series([X[c].value_counts().index[0]
            if X[c].dtype == np.dtype('O') else X[c].median() for c in X],
            index=X.columns)
        return self
    def transform(self, X, y=None):
        return X.fillna(self.fill)

feature_columns_to_use = ['Pclass','Sex','Age','Fare','Parch']
nonnumeric_columns = ['Sex']

#把train和test放一起处理变量
big_X = train_df[feature_columns_to_use].append(test_df[feature_columns_to_use])
big_X_imputed = DataFrameImputer().fit_transform(big_X)

#把字符型的变量转换为数值型，因为xgboost入参的变量必须是数值型的
le = LabelEncoder()
for feature in nonnumeric_columns:
    big_X_imputed[feature] = le.fit_transform(big_X_imputed[feature])

#xgboost的训练数据不能是dataframe格式，必须是矩阵形式（即matrix形式）
train_X = big_X_imputed[0:train_df.shape[0]].as_matrix()
test_X = big_X_imputed[train_df.shape[0]::].as_matrix()
train_y = train_df['Survived']

#xgboost参数设置（每一轮训练树的深度，学习率等参数）
gbm = xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05).fit(train_X, train_y)
predictions = gbm.predict(test_X)

#存储及输出结果
submission = pd.DataFrame({ 'PassengerId': test_df['PassengerId'],
                            'Survived': predictions })
submission.to_csv("submission.csv", index=False)

