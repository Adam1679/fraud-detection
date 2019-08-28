from sklearn.base import BaseEstimator, TransformerMixin
import json
import numpy as np
import pandas as pd
from itertools import product
class Email_Engineering(BaseEstimator, TransformerMixin):
    """
    对太多的域名进行了降纬分类，处理

    credit to ``https://www.kaggle.com/amirhmi/a-comprehensive-guide-to-get-0-9492``
    """

    def __init__(self, names):
        if not isinstance(names, list):
            self.names = list(names)
        else:
            self.names = names
        self.us_emails = {'gmail', 'net', 'edu'}

        with open("./ieee-fraud-detection/email.json") as f:
            self.emails = json.load(f)

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        for c in self.names:
            x[c + "_bin"] = x[c].map(self.emails)
            x[c + '_suffix'] = x[c].map(lambda x: str(x).split('.')[-1])
            x[c + '_suffix'] = x[c + '_suffix'].map(lambda x: x if str(x) not in self.us_emails else 'us')

        x['is_proton_mail'] = (x['P_emaildomain'] == 'protonmail.com') | (x['R_emaildomain'] == 'protonmail.com')

        return x


class Browser_Engineering(BaseEstimator, TransformerMixin):
    """
    对浏览器进行了处理

    credit to ``https://www.kaggle.com/amirhmi/a-comprehensive-guide-to-get-0-9492``
    """

    def __init__(self, name, verbose=1):
        self.name = name
        self.verbose = verbose

        with open("./ieee-fraud-detection/latest_browsers.txt") as f:
            self.latest_browser = set(map(str.strip, f.readlines()))

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        nan_mask = x[self.name].isnull()
        x['is_latest_browser'] = x[self.name].fillna("NaN")
        x['is_latest_browser'] = x['is_latest_browser'].map(lambda y: 1 if y in self.latest_browser else 0)
        x['is_latest_browser'] = x['is_latest_browser'].astype(np.int8)
        x.loc[nan_mask, 'is_latest_browser'] = np.nan
        if self.verbose:
            print(
                f"Summarize: # of 1 = {x['is_latest_browser'].sum()}, # of NaN = {x['is_latest_browser'].isnull().sum()}")
        return x


class Std_2var_Engineering(BaseEstimator, TransformerMixin):
    """
    双变量交互（std）

    credit to ``https://www.kaggle.com/amirhmi/a-comprehensive-guide-to-get-0-9492``
    """

    def __init__(self, numerical_features, categorical_features, verbose=1):
        self.n_feas = list(numerical_features)
        self.c_feas = list(categorical_features)
        self.verbose = verbose

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        for a, b in product(self.n_feas, self.c_feas):
            nan_mask = x[a].isnull() | x[b].isnull()
            name = a + "_to_std_" + b
            x[name] = x[a] / x.groupby([b])[a].transform('std')
            x.loc[nan_mask, name] = np.nan
            if self.verbose:
                print(f"Generate: {name}")
        return x


class Mean_2var_Engineering(BaseEstimator, TransformerMixin):
    """
    双变量交互（mean）
    credit to ``https://www.kaggle.com/amirhmi/a-comprehensive-guide-to-get-0-9492``
    """

    def __init__(self, numerical_features, categorical_features, verbose=1):
        self.n_feas = list(numerical_features)
        self.c_feas = list(categorical_features)
        self.verbose = verbose

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        for a, b in product(self.n_feas, self.c_feas):
            nan_mask = x[a].isnull() | x[b].isnull()
            name = a + "_to_mean_" + b
            x[name] = x[a] / x.groupby([b])[a].transform('mean')
            x.loc[nan_mask, name] = np.nan
            if self.verbose:
                print(f"Generate: {name}")
        return x


class Add_2var_Engineering(BaseEstimator, TransformerMixin):
    """
    双分类变量交互
    credit to ``https://www.kaggle.com/amirhmi/a-comprehensive-guide-to-get-0-9492``
    """

    def __init__(self, feature_pairs, verbose=1):
        self.pairs = list(feature_pairs)
        self.verbose = verbose

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        for feas in self.pairs:
            name = None
            if len(feas) == 2:
                a, b = feas
                nan_mask = x[a].isnull() | x[b].isnull()
                name = a + "_" + b
                x[name] = x[a].astype(str) + "_" + x[b].astype(str)
            elif len(feas) == 3:
                a, b, c = feas
                nan_mask = x[a].isnull() | x[b].isnull() | x[c].isnull()
                name = a + "_" + b + "_" + c
                x[name] = x[a].astype(str) + "_" + x[b].astype(str) + "_" + x[c].astype(str)

            x.loc[nan_mask, name] = np.nan
            if self.verbose:
                print(f"Generate: {name}")
        return x


class Count_Engineering(BaseEstimator, TransformerMixin):
    """
    添加分类变量的频率信息
    credit to ``https://www.kaggle.com/cdeotte/200-magical-models-santander-0-920``
    """

    def __init__(self, categorical_features, verbose=1):
        self.names = list(categorical_features)
        self.verbose = verbose
        self.counts = dict()

    def fit(self, x, y=None):
        for c in self.names:
            self.counts[c] = x[c].value_counts(dropna=False)
        return self

    def transform(self, x):
        for c in self.names:
            name = c + "_count"
            nan_mask = x[c].isnull()
            if not (c in self.counts):
                self.counts[c] = x[c].value_counts(dropna=False)

            if name in x.columns:
                name += "X"
            x[name] = x[c].map(self.counts[c])
            x.loc[nan_mask, name] = np.nan
            if self.verbose:
                print(f"Generate: {name}")
        return x


class Drop_Features(BaseEstimator, TransformerMixin):
    """
    删除一些的特征

    credit to ``https://www.kaggle.com/amirhmi/a-comprehensive-guide-to-get-0-9492``
    """

    def __init__(self, percentage, percentage_dup, verbose=1):
        self.perc = percentage
        self.perc_dup = percentage_dup
        self.verbose = verbose

    def fit(self, x, y=None):
        missing_values = x.isnull().sum() / len(x)
        missing_drop_cols = list(missing_values[missing_values > self.perc].keys())
        if "isFraud" in missing_drop_cols:
            missing_drop_cols.remove("isFraud")
        self.dropped_cols = missing_drop_cols
        duplicate_drop_cols = [col for col in x.columns if
                               x[col].value_counts(dropna=False, normalize=True).values[0] > self.perc_dup]
        if "isFraud" in duplicate_drop_cols:
            duplicate_drop_cols.remove("isFraud")
        self.dropped_cols.extend(duplicate_drop_cols)
        if self.verbose:
            print(f"Summarize: {len(missing_drop_cols)} columns have missing value(%) > {self.perc}")
            print(f"Summarize: {len(duplicate_drop_cols)} columns have duplicate value(%) > {self.perc_dup}")

        return self

    def transform(self, x):
        return x.drop(self.dropped_cols, axis=1)