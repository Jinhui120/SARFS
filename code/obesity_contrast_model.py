
#步骤二

#肥胖预测的对比模型
"""
1.需要把预处理+特征工程处理好的数据拿出来形成新的x和y
2.挑选10个主流机器学习模型进行对比
"""

import pandas as pd

obesity_prepared_data = pd.read_csv("./preprocess_obesity_dataset.csv")
print(obesity_prepared_data)
print(obesity_prepared_data.shape)#2111*16

obesity_prepared_data_x, obesity_prepared_data_y =  obesity_prepared_data.iloc[:, 0:15], obesity_prepared_data.iloc[:, -1]
print(obesity_prepared_data_x)#2111*15
print(obesity_prepared_data_y)#2111*1

#step_1：划分训练集和测试集
from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(obesity_prepared_data_x, obesity_prepared_data_y,
                                                    test_size=0.4, random_state=42)

#step_2：模型训练
"""
1.挑选出主流的机器学习模型用于对比分析，lr、sgd、svm、mlp、knn、etc、nb、dtree、rf、adaboost、xgboost、catboost、gbdt、lgbm
2.挑选出最优的模型用于后续的参数优化
3.在实验过程中需要加入交叉验证，防止模型过拟合
"""

#调用模型评估函数
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score  # 导入分类评价指标

#忽略两种警告
import warnings
from sklearn.exceptions import DataConversionWarning
# 忽略 DataConversionWarning 警告
warnings.filterwarnings("ignore", category=DataConversionWarning)


def evaluate_model(model, X, y):
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    pre = precision_score(y, y_pred, average='macro')  # 根据情况选择average参数
    rec = recall_score(y, y_pred, average='macro')
    f1 = f1_score(y, y_pred, average='macro')
    return acc, pre, rec, f1

#封装函数
from sklearn.model_selection import StratifiedKFold
import numpy as np

def cross_validation(model, train_x, train_y, test_x, test_y, n_splits, evaluate_model):

    """
    :param model: 模型的选择
    :param train_x: 训练集x
    :param train_y: 训练集y
    :param test_x: 测试集x
    :param test_y: 测试集y
    :param n_splits: 交叉验证中的折数
    :param evaluate_model: 评估函数
    """

    # 初始化五折交叉验证
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    # 存储每次交叉验证的评估指标
    acc_scores = []
    pre_scores = []
    rec_scores = []
    f1_scores = []

    for train_index, val_index in kfold.split(train_x, train_y):
        X_train_fold, X_val_fold = train_x.iloc[train_index], train_x.iloc[val_index]
        y_train_fold, y_val_fold = train_y.iloc[train_index], train_y.iloc[val_index]

        # 训练模型
        model.fit(X_train_fold, y_train_fold)

        # 评估模型
        acc, pre, rec, f1 = evaluate_model(model, X_val_fold, y_val_fold)

        # 存储评估指标
        acc_scores.append(acc)
        pre_scores.append(pre)
        rec_scores.append(rec)
        f1_scores.append(f1)

        # 打印每次交叉验证的评估指标
        print(f"本次交叉验证 - acc: {acc:.4f}, pre: {pre:.4f}, rec: {rec:.4f}, f1: {f1:.4f}")

    # 打印交叉验证的平均评估指标
    print(f"交叉验证平均 - acc: {np.mean(acc_scores):.4f}, pre: {np.mean(pre_scores):.4f},"
          f" rec: {np.mean(rec_scores):.4f}, f1: {np.mean(f1_scores):.4f}")

    # 在测试集上进行最终评估
    model.fit(train_x, train_y)
    acc, pre, rec, f1 = evaluate_model(model, test_x, test_y)
    print(f"测试集评估 - acc: {acc:.4f}, pre: {pre:.4f}, rec: {rec:.4f}, f1: {f1:.4f}")


"""
lr、sgd、svm、mlp、knn、etc、nb、dtree、rf、adaboost、xgboost、catboost、gbdt、lgbm
"""

#使用lr进行分类
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
# 忽略ConvergenceWarning警告
warnings.filterwarnings("ignore", category=ConvergenceWarning)
# 初始化逻辑回归模型
lr_model = LogisticRegression(random_state=42)
# 进行交叉验证和测试集评估
print("lr模型分类结果：")
cross_validation(lr_model, train_x, train_y, test_x, test_y, n_splits=5, evaluate_model=evaluate_model)

#使用sgd进行分类
from sklearn.linear_model import SGDClassifier
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
# 初始化SGDClassifier模型
sgd_model = SGDClassifier(random_state=42)
# 进行交叉验证和测试集评估
print("sgd模型分类结果：")
cross_validation(sgd_model, train_x, train_y, test_x, test_y, n_splits=5, evaluate_model=evaluate_model)

#使用mlp进行分类
from sklearn.neural_network import MLPClassifier
# 初始化MLPClassifier模型
mlp_model = MLPClassifier(random_state=42)
# 进行交叉验证和测试集评估
print("mlp模型分类结果：")
cross_validation(mlp_model, train_x, train_y, test_x, test_y, n_splits=5, evaluate_model=evaluate_model)

#使用knn进行分类
from sklearn.neighbors import KNeighborsClassifier
# 初始化KNeighborsClassifier模型
knn_model = KNeighborsClassifier(n_neighbors=10)
# 进行交叉验证和测试集评估
print("knn模型分类结果：")
cross_validation(knn_model, train_x, train_y, test_x, test_y, n_splits=5, evaluate_model=evaluate_model)

#使用etc进行分类
from sklearn.ensemble import ExtraTreesClassifier
# 初始化ExtraTreesClassifier模型
etc_model = ExtraTreesClassifier(random_state=42)
# 进行交叉验证和测试集评估
print("etc模型分类结果：")
cross_validation(etc_model, train_x, train_y, test_x, test_y, n_splits=5, evaluate_model=evaluate_model)

#使用nb进行分类
from sklearn.naive_bayes import GaussianNB
# 初始化GaussianNB模型
nb_model = GaussianNB()
# 进行交叉验证和测试集评估
print("nb模型分类结果：")
cross_validation(nb_model, train_x, train_y, test_x, test_y, n_splits=5, evaluate_model=evaluate_model)

#使用dtree进行分类
from sklearn.tree import DecisionTreeClassifier
# 初始化DecisionTreeClassifier模型
dtree_model = DecisionTreeClassifier(random_state=42)
# 进行交叉验证和测试集评估
print("dtree模型分类结果：")
cross_validation(dtree_model, train_x, train_y, test_x, test_y, n_splits=5, evaluate_model=evaluate_model)

#使用rf进行分类
from sklearn.ensemble import RandomForestClassifier
# 初始化RandomForestClassifier模型
rf_model = RandomForestClassifier(random_state=42)
# 进行交叉验证和测试集评估
print("rf模型分类结果：")
cross_validation(rf_model, train_x, train_y, test_x, test_y, n_splits=5, evaluate_model=evaluate_model)

#使用adaboost进行分类
from sklearn.ensemble import AdaBoostClassifier
# 初始化AdaBoostClassifier模型
adaboost_model = AdaBoostClassifier(random_state=42)
# 进行交叉验证和测试集评估
print("adaboost模型分类结果：")
cross_validation(adaboost_model, train_x, train_y, test_x, test_y, n_splits=5, evaluate_model=evaluate_model)

#使用xgboost进行分类
from xgboost import XGBClassifier
# 初始化XGBClassifier模型
xgb_model = XGBClassifier(random_state=42)
# 进行交叉验证和测试集评估
print("xgboost模型分类结果：")
cross_validation(xgb_model, train_x, train_y, test_x, test_y, n_splits=5, evaluate_model=evaluate_model)

# 使用catboost进行分类
from catboost import CatBoostClassifier
# 初始化CatBoostClassifier模型
catboost_model = CatBoostClassifier(random_state=42)
# 进行交叉验证和测试集评估
print("catboost模型分类结果：")
cross_validation(catboost_model, train_x, train_y, test_x, test_y, n_splits=5, evaluate_model=evaluate_model)

# 使用gbdt进行分类
from sklearn.ensemble import GradientBoostingClassifier
# 初始化GradientBoostingClassifier模型
gbdt_model = GradientBoostingClassifier(random_state=42)
# 进行交叉验证和测试集评估
print("gbdt模型分类结果：")
cross_validation(gbdt_model, train_x, train_y, test_x, test_y, n_splits=5, evaluate_model=evaluate_model)

# 使用lgbm进行分类
import lightgbm as lgb
# 初始化LGBMClassifier模型
lgbm_model = lgb.LGBMClassifier(random_state=42)
# 进行交叉验证和测试集评估
print("lgbm模型分类结果：")
cross_validation(lgbm_model, train_x, train_y, test_x, test_y, n_splits=5, evaluate_model=evaluate_model)

"""
最优模型为rf，下一步使用优化方法对参数进行优化
"""
