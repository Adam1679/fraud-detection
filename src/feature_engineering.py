#!/usr/bin/env python
# coding: utf-8

# In[172]:


import pandas as pd
import numpy as np
import lightgbm as lgb
import gc
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from src.feature.feature import *
from multiprocessing import Pool
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            if str(col_type)[:3] == 'int':
                df[col] = pd.to_numeric(df[col], downcast="integer")
            else:
                df[col] = pd.to_numeric(df[col], downcast="float")

    end_mem = df.memory_usage().sum() / 1024 ** 2
    if verbose:
        print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.
              format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


def load_data(file):
    return reduce_mem_usage(pd.read_csv(file, index_col='TransactionID'))


DIR = "./ieee-fraud-detection"

files = [f'{DIR}/train_transaction.csv',
         f'{DIR}/test_transaction.csv',
         f'{DIR}/train_identity.csv',
         f'{DIR}/test_identity.csv',
         f'{DIR}/sample_submission.csv']


with Pool() as pool:
    train_transaction, test_transaction, train_identity, test_identity, sample_submission = pool.map(load_data, files)

train = train_transaction.merge(train_identity, how='left', left_index=True, right_index=True)
test = test_transaction.merge(test_identity, how='left', left_index=True, right_index=True)
dataset = pd.concat([train.drop("isFraud", axis=1), test], ignore_index=True, sort=False)
del train_transaction, train_identity
del test_transaction, test_identity
gc.collect()

E1 = Email_Engineering(['P_emaildomain', 'R_emaildomain'])
train = E1.transform(train)
test = E1.transform(test)

B1 = Browser_Engineering("id_31", verbose=1)
train = B1.transform(train)
test = B1.transform(test)

D1 = Drop_Features(percentage=0.8, percentage_dup=0.9, verbose=1)
D1.fit(dataset)
train = D1.transform(train)
test = D1.transform(test)

M1 = Mean_2var_Engineering(numerical_features=["TransactionAmt", "id_02", "D15"],
                           categorical_features=['card1', 'card4', 'addr1'])
train = M1.transform(train)
test = M1.transform(test)

# In[180]:


A1 = Add_2var_Engineering(feature_pairs=[('card1', 'card4')])
train = A1.transform(train)
test = A1.transform(test)

# In[181]:


A2 = Add_2var_Engineering(feature_pairs=[('card1_card4', 'card3', 'card5')])
train = A2.transform(train)
test = A2.transform(test)
A3 = Add_2var_Engineering(feature_pairs=[('card1_card4_card3_card5', 'addr1', 'addr2')])
train = A3.transform(train)
test = A3.transform(test)

# In[182]:

# 未来数据？？？
train['TransactionAmt_check'] = np.where(train['TransactionAmt'].isin(test['TransactionAmt']), 1, 0)
test['TransactionAmt_check'] = np.where(test['TransactionAmt'].isin(train['TransactionAmt']), 1, 0)

# In[183]:


train['TransactionAmt'] = np.log1p(train['TransactionAmt'])
test['TransactionAmt'] = np.log1p(test['TransactionAmt'])

# In[184]:


C1 = Count_Engineering(categorical_features=['id_36'])
C2 = Count_Engineering(categorical_features=['id_01', 'id_31', 'id_35', 'id_36'])
C1.fit(dataset)
train = C1.transform(train)
test = C1.transform(test)
train = C2.transform(train)
test = C2.transform(test)

# In[185]:


# 看不懂的操作
for col in ['card1']:
    valid_card = pd.concat([train[[col]], test[[col]]])
    valid_card = valid_card[col].value_counts()
    valid_card = valid_card[valid_card > 2]
    valid_card = list(valid_card.index)

    train[col] = np.where(train[col].isin(test[col]), train[col], np.nan)
    test[col] = np.where(test[col].isin(train[col]), test[col], np.nan)

    train[col] = np.where(train[col].isin(valid_card), train[col], np.nan)
    test[col] = np.where(test[col].isin(valid_card), test[col], np.nan)

# In[186]:


numerical_columns = list(test.select_dtypes(exclude=['object']).columns)

train[numerical_columns] = train[numerical_columns].fillna(train[numerical_columns].median())
test[numerical_columns] = test[numerical_columns].fillna(train[numerical_columns].median())

# In[187]:


# 众数填充
categorical_columns = list(filter(lambda x: x not in numerical_columns, list(test.columns)))
train[categorical_columns] = train[categorical_columns].fillna(train[categorical_columns].mode())
test[categorical_columns] = test[categorical_columns].fillna(train[categorical_columns].mode())

# In[188]:


from sklearn.preprocessing import LabelEncoder

# 很有问题啊，这个假设的是有序分类变量，才能这样
for col in categorical_columns:
    le = LabelEncoder()
    le.fit(list(train[col].astype(str).values) + list(test[col].astype(str).values))
    train[col] = le.transform(list(train[col].astype(str).values))
    test[col] = le.transform(list(test[col].astype(str).values))

# # 2. 训练模型

# In[189]:


labels = train["isFraud"]
train.drop(["isFraud"], axis=1, inplace=True)

X_train, y_train = train, labels
del train, labels
gc.collect()


lgb_submission = sample_submission.copy()
lgb_submission['isFraud'] = 0
n_fold = 5
folds = KFold(n_fold)


for fold_n, (train_index, valid_index) in enumerate(folds.split(X_train)):
    print(fold_n)

    X_train_, X_valid = X_train.iloc[train_index], X_train.iloc[valid_index]
    y_train_, y_valid = y_train.iloc[train_index], y_train.iloc[valid_index]
    dtrain = lgb.Dataset(X_train, label=y_train)
    dvalid = lgb.Dataset(X_valid, label=y_valid)

    lgbclf = lgb.LGBMClassifier(
        num_leaves=512,
        n_estimators=512,
        max_depth=9,
        learning_rate=0.064,
        subsample=0.85,
        colsample_bytree=0.85,
        boosting_type="gbdt",
        reg_alpha=0.3,
        reg_lamdba=0.243
    )

    X_train_, X_valid = X_train.iloc[train_index], X_train.iloc[valid_index]
    y_train_, y_valid = y_train.iloc[train_index], y_train.iloc[valid_index]
    lgbclf.fit(X_train_, y_train_)

    del X_train_, y_train_

    print('finish train')
    pred = lgbclf.predict_proba(test)[:, 1]
    val = lgbclf.predict_proba(X_valid)[:, 1]
    print('finish pred')
    del lgbclf, X_valid
    print('ROC accuracy: {}'.format(roc_auc_score(y_valid, val)))
    del val, y_valid
    lgb_submission['isFraud'] = lgb_submission['isFraud'] + pred / n_fold
    del pred
    gc.collect()


lgb_submission.insert(0, "TransactionID", np.arange(3663549, 3663549 + 506691))
lgb_submission.to_csv('prediction.csv', index=False)
