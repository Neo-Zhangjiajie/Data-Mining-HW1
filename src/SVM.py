import numpy as np
from src.dataset import load_data

class LinearSVM:
    def __init__(self, learning_rate=0.0001, lambda_param=0.01, n_iters=1000):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        # 将标签转换为-1和1
        y_ = np.where(y <= 0, -1, 1)

        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # 梯度下降优化
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.weights) - self.bias) >= 1
                if condition:
                    self.weights -= self.learning_rate * (2 * self.lambda_param * self.weights)
                else:
                    self.weights -= self.learning_rate * (2 * self.lambda_param * self.weights - np.dot(x_i, y_[idx]))
                    self.bias -= self.learning_rate * y_[idx]

    def predict(self, X):
        linear_output = np.dot(X, self.weights) - self.bias
        return np.sign(linear_output)
def accuracy_score(y_true, y_pred):
    return np.mean(y_true == y_pred)

def precision_recall_f1(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == -1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == -1))

    precision = tp / (tp + fp) if (tp + fp) != 0 else 0
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

    return precision, recall, f1

def roc_auc_score(y_true, y_scores):
    # 这里我们使用了简化的方法来计算AUC，基于排序的方法
    n_pos = np.sum(y_true == 1)
    n_neg = len(y_true) - n_pos
    rank = np.argsort(y_scores)
    rank_pos = np.sum(rank[y_true == 1]) - n_pos * (n_pos - 1) / 2
    auc = rank_pos / (n_pos * n_neg)
    return auc
# 使用示例
def train_linear_svm():
    X_train, X_test, y_train, y_test = load_data()  # 加载数据

    svm_model = LinearSVM()
    svm_model.fit(X_train, y_train)
    predictions = svm_model.predict(X_test)
    # 计算准确率、F1分数和AUC
    acc = accuracy_score(y_test, predictions)
    precision, recall, f1 = precision_recall_f1(y_test, predictions)

    # 使用SVM的决策函数值来计算AUC
    decision_scores = np.dot(X_test, svm_model.weights) - svm_model.bias
    auc = roc_auc_score(y_test, decision_scores)

    print(f"Accuracy: {acc}")
    print(f"F1 Score: {f1}")
    print(f"AUC: {auc}")
