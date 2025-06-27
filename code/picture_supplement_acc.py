
#做以下实验补充（画图不够多，加入美观）
"""
1.做三个混淆矩阵的图（ETC、RF和SARFS）
2.做ROC曲线的图（mlp、etc、Dtree、rf、pso、rf、bo、rf、sarfs）
3.做SSA优化RF过程的准确率的图
"""

#做SSA优化RF过程的准确率的图

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
#使用rf进行分类
from sklearn.ensemble import RandomForestClassifier

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


# 麻雀优化算法
import matplotlib.pyplot as plt


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

    # 新增：记录每次迭代的最优准确率
    accuracy_history = []

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

        # 新增：记录当前迭代的最优准确率
        best_fitness = np.max(fitness_values)
        accuracy_history.append(best_fitness)

        # 打印进度（可选）
        if (t + 1) % 10 == 0:
            print(f"迭代 {t + 1}/{max_iter}, 最优准确率: {best_fitness:.4f}")

    # 找到最优参数
    best_index = np.argmax(fitness_values)
    best_params = population[best_index]
    n_estimators = int(best_params[0])
    max_depth = int(best_params[1])

    # 新增：绘制准确率变化图
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_iter + 1), accuracy_history, 'o-', color='blue')
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('测试集准确率', fontsize=12)
    plt.title('SSA优化RF过程中准确率的变化', fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('ssa_optimization_accuracy.jpg', dpi=600, bbox_inches='tight')
    plt.show()

    return n_estimators, max_depth


# 使用SSA优化RF参数
n_estimators, max_depth = ssa_optimize_rf(train_x, train_y, test_x, test_y)
print(f"最优参数 - n_estimators: {n_estimators}, max_depth: {max_depth}")

# 使用优化后的参数创建RF模型
rf_model_optimized = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)

# 进行交叉验证和测试集评估
print("优化后的rf模型分类结果：")
cross_validation(rf_model_optimized, train_x, train_y, test_x, test_y, n_splits=5, evaluate_model=evaluate_model)
