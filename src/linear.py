import numpy as np
from sklearn.metrics import roc_auc_score

class LinearRegression:
    def __init__(self):
        self.coefficients = None

    def fit(self, X, y):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]  # 添加截距项
        self.coefficients = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

    def predict(self, X):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]  # 添加截距项
        return X_b.dot(self.coefficients)

class LinearRegressionClassifier:
    def __init__(self, threshold=0.5):
        self.linear_regression = LinearRegression()
        self.threshold = threshold

    def fit(self, X, y):
        self.linear_regression.fit(X, y)

    def predict(self, X):
        continuous_pred = self.linear_regression.predict(X)
        return (continuous_pred > self.threshold).astype(int)

    def predict_proba(self, X):
        # 使用sigmoid函数将线性回归输出转换为概率
        continuous_pred = self.linear_regression.predict(X)
        return 1 / (1 + np.exp(-continuous_pred))

def accuracy_score(y_true, y_pred):
    return np.mean(y_true == y_pred)

def precision_recall_f1(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    precision = tp / (tp + fp) if (tp + fp) != 0 else 0
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0

    return precision, recall, f1

def get_linear_result(X_train, y_train, X_test, y_test):
    # 训练线性回归分类器
    model = LinearRegressionClassifier()
    model.fit(X_train, y_train)

    # 预测并计算各项指标
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)
    precision, recall, f1 = precision_recall_f1(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    return acc,precision,recall,f1,auc
    print(f"Accuracy: {acc}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    print(f"AUC: {auc}")
