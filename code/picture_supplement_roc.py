

#做以下实验补充（画图不够多，加入美观）
"""
1.做三个混淆矩阵的图（ETC、RF和SARFS）
2.做ROC曲线的图（mlp、etc、Dtree、rf、pso、rf、bo、rf、sarfs）
3.做SSA优化RF过程的准确率的图
"""

#step_1：画ROC曲线
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

#使用mlp进行分类
from sklearn.neural_network import MLPClassifier
# 初始化MLPClassifier模型
mlp_model = MLPClassifier(random_state=42)
# 进行交叉验证和测试集评估
print("mlp模型分类结果：")
cross_validation(mlp_model, train_x, train_y, test_x, test_y, n_splits=5, evaluate_model=evaluate_model)

#使用etc进行分类
from sklearn.ensemble import ExtraTreesClassifier
# 初始化ExtraTreesClassifier模型
etc_model = ExtraTreesClassifier(random_state=42)
# 进行交叉验证和测试集评估
print("etc模型分类结果：")
cross_validation(etc_model, train_x, train_y, test_x, test_y, n_splits=5, evaluate_model=evaluate_model)

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

##使用PSO来优化rf模型参数
import pyswarms as ps
# 导入粒子群优化算法（PSO）的库pyswarms

# 定义适应度函数
def fitness_function(params):
    # 将第一个参数转换为整数，作为随机森林中决策树的数量
    n_estimators = int(params[0][0])
    # 将第二个参数转换为整数，作为随机森林中决策树的最大深度
    max_depth = int(params[0][1])
    # 创建随机森林分类器对象，设置决策树数量、最大深度和随机种子
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    # 创建分层K折交叉验证对象，设置折数、是否打乱数据和随机种子
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    acc_scores = []
    # 对每一次交叉验证的训练集和验证集索引进行遍历
    for train_index, val_index in kfold.split(train_x, train_y):
        # 根据索引获取训练集的特征和标签
        X_train_fold, X_val_fold = train_x.iloc[train_index], train_x.iloc[val_index]
        # 根据索引获取验证集的特征和标签
        y_train_fold, y_val_fold = train_y.iloc[train_index], train_y.iloc[val_index]
        # 使用训练集数据训练随机森林模型
        model.fit(X_train_fold, y_train_fold)
        # 使用训练好的模型对验证集进行预测
        y_pred = model.predict(X_val_fold)
        # 计算预测结果与真实标签的准确率
        acc = accuracy_score(y_val_fold, y_pred)
        # 将准确率添加到列表中
        acc_scores.append(acc)
    # 返回1减去平均准确率，作为适应度函数值，因为要最小化这个值
    return 1 - np.mean(acc_scores)

# 定义参数边界
dimensions = 2
# 定义参数的下限，分别是决策树数量的下限和最大深度的下限
bounds = (np.array([100, 1]), np.array([300, 10]))

# 初始化PSO优化器
options = {'c1': 0.5, 'c2': 0.4, 'w': 0.6}
# 创建粒子群优化器对象，设置粒子数量、参数维度、优化选项和参数边界
optimizer = ps.single.GlobalBestPSO(n_particles=20, dimensions=dimensions, options=options, bounds=bounds)

# 进行优化
cost, pos = optimizer.optimize(fitness_function, iters=20)
# 使用PSO优化器对适应度函数进行优化，迭代10次，返回最小化的适应度值和最优参数位置

# 使用优化后的参数创建RF模型
n_estimators = int(pos[0])
# 从最优参数位置中获取决策树数量，并转换为整数
max_depth = int(pos[1])
# 从最优参数位置中获取最大深度，并转换为整数
# 创建优化后的随机森林分类器对象
optimized_rf_model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)

# 进行交叉验证和测试集评估
print("优化后的rf模型分类结果：")
# 调用交叉验证函数，传入优化后的模型、训练集、测试集等参数进行评估
cross_validation(optimized_rf_model, train_x, train_y, test_x, test_y, n_splits=5, evaluate_model=evaluate_model)

"""
# 初始化PSO优化器
options = {'c1': 0.5, 'c2': 0.4, 'w': 0.6}
n_particles=20
测试集评估 - acc: 0.9846, pre: 0.9841, rec: 0.9844, f1: 0.9842
"""

# 麻雀优化算法
def ssa_optimize_rf(train_x, train_y, test_x, test_y, n=30, max_iter=50):
    # 定义参数范围
    param_ranges = [
        (10, 200),  # n_estimators
        (2, 20)  # max_depth
    ]
    dim = len(param_ranges)

    # 初始化麻雀种群
    population = np.zeros((n, dim))
    for i in range(n):
        for j in range(dim):
            population[i, j] = np.random.randint(param_ranges[j][0], param_ranges[j][1])

    # 计算适应度
    def fitness(params):
        n_estimators = int(params[0])
        max_depth = int(params[1])
        model = RandomForestClassifier(n_estimators=n_estimators,
                                       max_depth=max_depth,
                                       random_state=42)
        model.fit(train_x, train_y)
        y_pred = model.predict(test_x)
        return accuracy_score(test_y, y_pred)

    fitness_values = np.array([fitness(population[i]) for i in range(n)])

    # 开始迭代
    for t in range(max_iter):
        # 发现者更新
        alpha = 0.2
        r2 = np.random.rand()
        if r2 < 0.8:
            for i in range(int(0.2 * n)):
                for j in range(dim):
                    population[i, j] = population[i, j] * np.exp(-i / (alpha * max_iter))
        else:
            for i in range(int(0.2 * n)):
                for j in range(dim):
                    population[i, j] = population[i, j] + np.random.randn()

        # 追随者更新
        for i in range(int(0.2 * n), n):
            best_index = np.argmax(fitness_values)
            if i > n / 2:
                for j in range(dim):
                    population[i, j] = np.random.randn() * np.exp(
                        (population[best_index, j] - population[i, j]) / (i - n / 2) ** 2)
            else:
                A = np.random.choice([-1, 1], dim)
                A_plus = np.linalg.pinv(A.reshape(-1, 1))
                for j in range(dim):
                    population[i, j] = population[best_index, j] + np.abs(
                        population[i, j] - population[best_index, j]) * A_plus[0, j] * np.random.randn()

        # 警戒者更新
        s = 0.1
        for i in range(int(0.2 * n)):
            if fitness_values[i] > np.mean(fitness_values):
                for j in range(dim):
                    population[i, j] = population[i, j] + s * np.random.randn()
            else:
                for j in range(dim):
                    population[i, j] = population[i, j] + np.random.randn() * (
                            np.abs(population[i, j] - population[np.argmax(fitness_values), j]))

        # 边界处理
        for i in range(n):
            for j in range(dim):
                if population[i, j] < param_ranges[j][0]:
                    population[i, j] = param_ranges[j][0]
                elif population[i, j] > param_ranges[j][1]:
                    population[i, j] = param_ranges[j][1]

        # 计算新的适应度
        fitness_values = np.array([fitness(population[i]) for i in range(n)])

    # 找到最优参数
    best_index = np.argmax(fitness_values)
    best_params = population[best_index]
    n_estimators = int(best_params[0])
    max_depth = int(best_params[1])
    return n_estimators, max_depth


# 使用SSA优化RF参数
n_estimators, max_depth = ssa_optimize_rf(train_x, train_y, test_x, test_y)
print(f"最优参数 - n_estimators: {n_estimators}, max_depth: {max_depth}")

# 使用优化后的参数创建RF模型
rf_model_optimized = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)

# 进行交叉验证和测试集评估
print("优化后的rf模型分类结果：")
cross_validation(rf_model_optimized, train_x, train_y, test_x, test_y, n_splits=5, evaluate_model=evaluate_model)

"""
1.最优参数 - n_estimators: 104, max_depth: 15
2.测试集评估 - acc: 0.9858, pre: 0.9853, rec: 0.9857, f1: 0.9853
"""

from bayes_opt import BayesianOptimization

# 定义贝叶斯优化的目标函数
def rf_cv(n_estimators, max_depth):
    n_estimators = int(n_estimators)
    max_depth = int(max_depth)
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    acc_scores = []
    for train_index, val_index in kfold.split(train_x, train_y):
        X_train_fold, X_val_fold = train_x.iloc[train_index], train_x.iloc[val_index]
        y_train_fold, y_val_fold = train_y.iloc[train_index], train_y.iloc[val_index]
        model.fit(X_train_fold, y_train_fold)
        y_pred = model.predict(X_val_fold)
        acc = accuracy_score(y_val_fold, y_pred)
        acc_scores.append(acc)
    return np.mean(acc_scores)

# 定义参数搜索空间
pbounds = {'n_estimators': (10, 200),
           'max_depth': (1, 50)}

# 初始化贝叶斯优化器
optimizer = BayesianOptimization(
    f=rf_cv,
    pbounds=pbounds,
    random_state=42,
)

# 进行贝叶斯优化
optimizer.maximize(
    init_points=5,
    n_iter=20,
)

# 获取最优参数
best_params = optimizer.max['params']
best_n_estimators = int(best_params['n_estimators'])
best_max_depth = int(best_params['max_depth'])

# 输出优化后的参数
print(f"优化后的 n_estimators 参数值: {best_n_estimators}")
print(f"优化后的 max_depth 参数值: {best_max_depth}")

# 使用最优参数初始化随机森林模型
best_rf_model = RandomForestClassifier(n_estimators=best_n_estimators, max_depth=best_max_depth, random_state=42)

# 进行交叉验证和测试集评估
print("使用贝叶斯优化后的rf模型分类结果：")
cross_validation(best_rf_model, train_x, train_y, test_x, test_y, n_splits=5, evaluate_model=evaluate_model)

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import numpy as np

# 假设 test_y 是 numpy.ndarray 类型
unique_classes = np.sort(np.unique(test_y))
# 将标签二值化（7 分类 → (n_samples, 7) 的one-hot矩阵）
y_test_binarized = label_binarize(test_y, classes=unique_classes)  # unique_classes 是 [0,1,2,3,4,5,6]

# 存储模型名称和模型对象（保持你的 models_for_roc 定义）
models_for_roc = [
    ("MLP", mlp_model),
    ("ETC", etc_model),
    ("Dtree", dtree_model),
    ("RF", rf_model),
    ("PSO-RF", optimized_rf_model),
    ("SSA-RF", rf_model_optimized),
    ("BO-RF", best_rf_model)
]

plt.figure(figsize=(8, 6), dpi=600)

# 遍历模型
for model_idx, (model_name, model) in enumerate(models_for_roc):
    # 获取预测概率：(n_samples, 7)
    y_pred_proba = model.predict_proba(test_x)

    # 计算每个类别的 FPR、TPR
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for class_idx in range(len(unique_classes)):
        fpr[class_idx], tpr[class_idx], _ = roc_curve(y_test_binarized[:, class_idx], y_pred_proba[:, class_idx])
        roc_auc[class_idx] = auc(fpr[class_idx], tpr[class_idx])

    # 计算宏平均 ROC 曲线（对每个类别的 FPR、TPR 取平均）
    all_fpr = np.unique(np.concatenate([fpr[class_idx] for class_idx in range(len(unique_classes))]))
    mean_tpr = np.zeros_like(all_fpr)
    for class_idx in range(len(unique_classes)):
        mean_tpr += np.interp(all_fpr, fpr[class_idx], tpr[class_idx])
    mean_tpr /= len(unique_classes)

    # 计算宏平均 AUC
    macro_auc = auc(all_fpr, mean_tpr)

    # 定义模型颜色和线型（添加这部分代码）
    model_colors = ['#3357FF', 'g', '#FF5733', 'c', 'm', 'r', 'y']  # 7种颜色对应7个模型
    model_linestyles = ['-', '--', '-.', ':', '-', '--', '-.']  # 线型列表


    # 绘制宏平均 ROC 曲线
    plt.plot(
        all_fpr, mean_tpr,
        color=model_colors[model_idx % len(model_colors)],
        linestyle=model_linestyles[model_idx % len(model_linestyles)],
        label=f'{model_name}'
    )

# 绘制随机猜测线
plt.plot([0, 1], [0, 1], color='gray')

# 优化显示
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.legend(loc='lower right', fontsize=10)
plt.grid(True)
plt.tight_layout()
plt.savefig('roc_curves_macro.jpg', dpi=600, bbox_inches='tight', pad_inches=0)
plt.show()

