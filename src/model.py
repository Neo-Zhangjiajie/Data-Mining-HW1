from sklearn.svm import SVC  # 导入支持向量分类类
from sklearn.tree import DecisionTreeClassifier, plot_tree  # 导入决策树分类类
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier  # 导入随机森林分类类
from sklearn.linear_model import LogisticRegression  # 导入逻辑回归类
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score  # 导入分类评估指标
import numpy as np  # 导入numpy库
from src.dataset import load_data  # 假设这个函数用于获取数据集
from src.linear import get_linear_result
import pandas as pd  # 导入pandas库
import pdb

# 训练模型的函数不需要改变，只是模型类变了
from sklearn.metrics import roc_auc_score  # 导入AUC计算函数


def train_model(X_train, y_train, X_test, y_test, model_class, is_svc=False):
    # 如果是SVC模型，需要设置probability=True以输出概率
    if is_svc:
        model = model_class(probability=True)
    else:
        model = model_class()
        # model = model_class(max_depth=3)

    model.fit(X_train, y_train)
    # plt.figure(figsize=(20,10))  # 设定画布大小
    # plot_tree(model, filled=True, feature_names=["age","default","balance","housing","loan","day","duration","campaign","pdays","previous","job_admin.","job_blue-collar","job_entrepreneur","job_housemaid","job_management","job_retired","job_self-employed","job_services","job_student","job_technician","job_unemployed","job_unknown","marital_divorced","marital_married","marital_single","education_primary","education_secondary","education_tertiary","education_unknown","contact_cellular","contact_telephone","contact_unknown","month_apr","month_aug","month_dec","month_feb","month_jan","month_jul","month_jun","month_mar","month_may","month_nov","month_oct","month_sep","poutcome_failure","poutcome_other","poutcome_success","poutcome_unknown"], class_names=["No Deposit", "Deposit"], rounded=True)
    # plt.show()
    y_pred = model.predict(X_test)
    # 用于AUC计算的概率预测
    if is_svc:
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(
            X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    auc = roc_auc_score(y_test, y_prob)  # 计算AUC
    return accuracy, precision, recall, f1, auc


def train():
    results = {"Metrics": [], "SVM": [], "Decision Tree": [], "Random Forest": [],
               "Logistic Regression": [],"Linear Regression Classification":[]}
    X_train, X_test, y_train, y_test = load_data()
    results["Metrics"] = ["ACC","PRE","RECALL","F1","AUC"]
    '''
    # SVM
    results["SVM"].extend(train_model(X_train, y_train, X_test, y_test, SVC, is_svc=True))

    # Decision Tree
    results["Decision Tree"].extend(train_model(X_train, y_train, X_test, y_test, DecisionTreeClassifier))

    # Random Forest
    results["Random Forest"].extend(train_model(X_train, y_train, X_test, y_test, RandomForestClassifier))

    # Logistic Regression
    results["Logistic Regression"].extend(train_model(X_train, y_train, X_test, y_test, LogisticRegression))
    '''
    results["Decision Tree"].extend(train_model(X_train, y_train, X_test, y_test, DecisionTreeClassifier))
    print(results)
    # 将结果转换为DataFrame并打印
    #results_df = pd.DataFrame(results)
    #print(results_df)
    #return results_df
