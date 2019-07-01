"""
Helper functions for categorical encodings
"""
from pandas.api.types import is_string_dtype, is_numeric_dtype
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction import FeatureHasher
from sklearn.model_selection import KFold
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

def kfold_target_encoder(train, test, cols_encode, target, folds=10):
    """
    Mean regularized target encoding based on kfold
    """
    train_new = train.copy()
    test_new = test.copy()
    kf = KFold(n_splits=folds, random_state=1)
    for col in cols_encode:
        global_mean = train_new[target].mean()
        for train_index, test_index in kf.split(train):
            mean_target = train_new.iloc[train_index].groupby(col)[target].mean()
            train_new.loc[test_index, col + "_mean_enc"] = train_new.loc[test_index, col].map(mean_target)
        train_new[col + "_mean_enc"].fillna(global_mean, inplace=True)
        # making test encoding using full training data
        col_mean = train_new.groupby(col)[target].mean()
        test_new[col + "_mean_enc"] = test_new[col].map(col_mean)
        test_new[col + "_mean_enc"].fillna(global_mean, inplace=True)
    
    # filtering only mean enc cols
    train_new = train_new.filter(like="mean_enc", axis=1)
    test_new = test_new.filter(like="mean_enc", axis=1)
    return train_new, test_new
        
def catboost_target_encoder(train, test, cols_encode, target):
    """
    Encoding based on ordering principle
    """
    train_new = train.copy()
    test_new = test.copy()
    for column in cols_encode:
        global_mean = train[target].mean()
        cumulative_sum = train.groupby(column)[target].cumsum() - train[target]
        cumulative_count = train.groupby(column).cumcount()
        train_new[column + "_cat_mean_enc"] = cumulative_sum/cumulative_count
        train_new[column + "_cat_mean_enc"].fillna(global_mean, inplace=True)
        # making test encoding using full training data
        col_mean = train_new.groupby(column).mean()[column + "_cat_mean_enc"]  #
        test_new[column + "_cat_mean_enc"] = test[column].map(col_mean)
        test_new[column + "_cat_mean_enc"].fillna(global_mean, inplace=True)
    # filtering only mean enc cols
    train_new = train_new.filter(like="cat_mean_enc", axis=1)
    test_new = test_new.filter(like="cat_mean_enc", axis=1)
    return train_new, test_new

def one_hot_encoder(train, test, cols_encode, target=None):
    """ one hot encoding"""
    ohc_enc = OneHotEncoder(handle_unknown='ignore')
    ohc_enc.fit(train[cols_encode])
    train_ohc = ohc_enc.transform(train[cols_encode])
    test_ohc = ohc_enc.transform(test[cols_encode])
    return train_ohc, test_ohc
    
def label_encoder(train, test, cols_encode=None, target=None):
    """
    Code borrowed from fast.ai and is tweaked a little.
    Convert columns in a training and test dataframe into numeric labels 
    """
    train_new = train.drop(target, axis=1).copy()
    test_new = test.drop(target, axis=1).copy()
    
    for n,c in train_new.items():
        if is_string_dtype(c) or n in cols_encode : train_new[n] = c.astype('category').cat.as_ordered()
    
    if test_new is not None:
        for n,c in test_new.items():
            if (n in train_new.columns) and (train_new[n].dtype.name=='category'):
                test_new[n] = pd.Categorical(c, categories=train_new[n].cat.categories, ordered=True)
            
    cols = list(train_new.columns[train_new.dtypes == 'category'])
    for c in cols:
        train_new[c] = train_new[c].astype('category').cat.codes
        if test_new is not None: test_new[c] = test_new[c].astype('category').cat.codes
    return train_new, test_new

def hash_encoder(train, test, cols_encode, target=None, n_features=10):
    """hash encoder"""
    h = FeatureHasher(n_features=n_features, input_type="string")
    for col_encode in cols_encode:
        h.fit(train[col_encode])
        train_hash = h.transform(train[col_encode])
        test_hash = h.transform(test[col_encode])
    return train_hash, test_hash


def fitmodel_and_auc_score(encoder, train, test, cols_encode, target, **kwargs):
    """
    Fits and returns scores of a random forest model. Uses ROCAUC as scoring metric
    """
    model = RandomForestClassifier(n_estimators=500,
                                   n_jobs=-1, 
                                   class_weight="balanced",
                                   max_depth=10)
    if encoder:
        train_encoder, test_encoder = encoder(train, test, cols_encode=cols_encode, target=target)
    else:
        train_encoder, test_encoder = train.drop(target, axis=1), test.drop(target, axis=1)
    model.fit(train_encoder, train[target])
    train_score = roc_auc_score(train[target], model.predict(train_encoder))
    valid_score = roc_auc_score(test[target], model.predict(test_encoder))
    return train_score, valid_score
