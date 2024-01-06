from src.model import train
from src.SVM import train_linear_svm
import pandas as pd
#train_linear_svm()
# 调用train_all函数，传入想要训练的数据集名称列表
train().to_excel("results.xlsx")