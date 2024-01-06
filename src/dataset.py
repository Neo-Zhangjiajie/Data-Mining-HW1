import pandas as pd
import os

def get_():
    base_root = "data/split"
    train_path = os.path.join(base_root,  "train.csv")
    dev_path = os.path.join(base_root,  "dev.csv")
    test_path = os.path.join(base_root,  "test.csv")
    train = pd.read_csv(train_path)
    dev = pd.read_csv(dev_path)
    test = pd.read_csv(test_path)
    return train, dev, test

def _load_data(dataframe):
    X = dataframe.iloc[:, :-1].values  # 除了最后一列，其余都是特征
    y = dataframe.iloc[:, -1].values  # 最后一列是目标变量
    return X, y

def load_data():
    train, dev, test = get_()
    X_train,y_train = _load_data(train)
    X_test,y_test = _load_data(test)
    return X_train, X_test, y_train, y_test

def load_feature():
    train, dev, test = get_()
    return train.columns.values.tolist()[:-1]