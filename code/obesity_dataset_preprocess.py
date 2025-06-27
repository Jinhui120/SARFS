
#步骤一


#obesity_prediction_model

"""
本次实验的主要思路是在模型优化方面，在做特征工程的时候可能不会做的很多
主要是想用优化方法来改进模型的精度
"""

##step_1特征工程
"""
1.数据介绍
2.缺失值处理
3.异常值处理
"""

import pandas as pd

# obesity_raw_data = pd.read_csv('./BERT_KDBiLSTM/ObesityDataSet_raw_and_data_sinthetic.csv')
obesity_raw_data = pd.read_csv("./ObesityDataSet_raw_and_data_sinthetic.csv")
print(obesity_raw_data.head())

#将所有数据集显示全
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
print(obesity_raw_data)
print(obesity_raw_data.shape)#(2111, 17)

#特征工程
##特征重命名
obesity_raw_data.columns = ['gender', 'age', 'height', 'weight', 'family_history_with_overweight',
                            'favc', 'fcvc', 'ncp', 'caec', 'smoke', 'ch2o', 'scc', 'faf', 'tue',
                            'calc', 'mtrans', 'no_obesity']
"""
gender:性别；age：年龄；height：身高；weight：体重；
family_history_with_overweight：家族是否有肥胖史；favc：是否经常吃高热量食物；fcvc：吃饭时是否经常吃蔬菜
ncp：一天吃几顿饭；caec：两餐之间是否吃其它零食；smoke：吸烟史；ch2o：每日饮水量
scc：是否每天监测卡路里；faf：多久锻炼一次；tue：使用电子设备次数
calc：多久喝一次酒；mtrans：经常使用什么交通工具出行；no_obesity：肥胖等级
"""
print(obesity_raw_data)

##显示数据类型
print(obesity_raw_data.dtypes)
##筛选出数据类型为object的列
object_columns = obesity_raw_data.select_dtypes(include=['object'])
# 统计每个object列的不同值的数量
unique_counts = object_columns.nunique()
print("数据类型为object的列的不同值的数量：")
print(unique_counts)

##特征编码（将object类型的特征编码为数值类型）

gender_dict = {'Male':0, 'Female':1}
obesity_raw_data['gender'] = obesity_raw_data['gender'].map(gender_dict)

family_history_with_overweight_dict = {'yes':0, 'no':1}
obesity_raw_data['family_history_with_overweight'] = obesity_raw_data['family_history_with_overweight'].map(family_history_with_overweight_dict)

favc_dict = {'yes':0, 'no':1}
obesity_raw_data['favc'] = obesity_raw_data['favc'].map(favc_dict)

caec_dict = {'Always':0, 'Frequently':1, 'no':2, 'Sometimes':3}
obesity_raw_data['caec'] = obesity_raw_data['caec'].map(caec_dict)

smoke_dict = {'yes':0, 'no':1}
obesity_raw_data['smoke'] = obesity_raw_data['smoke'].map(smoke_dict)

scc_dict = {'yes':0, 'no':1}
obesity_raw_data['scc'] = obesity_raw_data['scc'].map(scc_dict)

calc_dict = {'no':0, 'Sometimes':1, 'Frequently':2, 'Always':3}
obesity_raw_data['calc'] = obesity_raw_data['calc'].map(calc_dict)

mtrans_dict = {'Automobile':0, 'Bike':1, 'Motorbike':2, 'Public_Transportation':3, 'Walking':4}
obesity_raw_data['mtrans'] = obesity_raw_data['mtrans'].map(mtrans_dict)

no_obesity_dict = {'Insufficient_Weight':0, 'Normal_Weight':1, 'Obesity_Type_I':2,
                   'Obesity_Type_II':3, 'Obesity_Type_III':4, 'Overweight_Level_I':5, 'Overweight_Level_II':6}
obesity_raw_data['no_obesity'] = obesity_raw_data['no_obesity'].map(no_obesity_dict)

##重新检查数据
print(obesity_raw_data.shape)#(2111, 17)
print(obesity_raw_data.dtypes)

##需要将某些变量进行转化
"""
float64——>int
age:按照四舍五入；height和weight按照保留两位小数；
fcvc按照四舍五入；ncp按照四舍五入；ch2o按照四舍五入
faf按照四舍五入；tue按照四舍五入
"""

# age 列四舍五入并转换为整数
obesity_raw_data['age'] = obesity_raw_data['age'].round().astype('int64')
#calc转化为整型
obesity_raw_data['calc'] = obesity_raw_data['calc'].astype('int64')
# height 和 weight 保留两位小数
obesity_raw_data['height'] = obesity_raw_data['height'].round(2)
obesity_raw_data['weight'] = obesity_raw_data['weight'].round(2)

# fcvc、ncp、ch2o、faf 四舍五入并转换为整数
for col in ['fcvc', 'ncp', 'ch2o', 'faf', 'tue']:
    obesity_raw_data[col] = obesity_raw_data[col].round().astype('int64')

print(obesity_raw_data)
print(obesity_raw_data.dtypes)

#异常值处理
"""
1.缺失值填充（没有空缺值）
2.异常值处理
"""
# 先查看缺失值，在考虑用什么方法进行填充
def miss_data_count(data):
    # 统计每列的缺失值数量，并按降序排序
    miss_number = data.isnull().sum().sort_values(ascending=False)
    # 计算每列的缺失值比例，并按降序排序
    miss_percent = (data.isnull().sum() / data.count()).sort_values(ascending=False)
    # 将缺失值数量和缺失值比例合并为一个 DataFrame
    miss_values = pd.concat([miss_number, miss_percent], axis=1, keys=['miss_number', 'miss_percent'])
    return miss_values
print(miss_data_count(obesity_raw_data))

# #画缺失值的矩阵图
# import missingno as msno
# import matplotlib.pyplot as plt
# msno.matrix(obesity_raw_data, labels=True) # 矩阵
# plt.savefig('obesity_raw_data简略缺失值.jpg', dpi=600, bbox_inches='tight',pad_inches=0)
# plt.show()

##经过检查，没有缺失值，所以不需要处理缺失值情况
##一般这类数据不存在异常值，所以不进行判断异常值

##做特征衍生
"""
根据身高和体重做出一列新的特征bmi并放在倒数第二列
"""

# 计算BMI
obesity_raw_data['bmi'] = obesity_raw_data['weight'] / (obesity_raw_data['height']  ** 2)
obesity_raw_data['bmi'] = obesity_raw_data['bmi'].round(2)

# 将bmi特征移动到倒数第二列
last_column = obesity_raw_data.pop('bmi')
obesity_raw_data.insert(len(obesity_raw_data.columns) - 1, 'bmi', last_column)

print(obesity_raw_data)
print(obesity_raw_data.shape)
print(obesity_raw_data.dtypes)

#特征筛选
"""
1.先把原始数据划分为x和y
2.画变量x之间的相关性热力图，说明数据之间具有相关性
3.选用使用Lasso、户信息法（MIC）、递归特征消除法（RFE）
4.评价指标选用acc、pre、rec、f1
"""

obesity_raw_data_x = obesity_raw_data.iloc[:, :-1]
obesity_raw_data_y = obesity_raw_data.iloc[:, [-1]]
print(obesity_raw_data_x)#shape2111*17
print(obesity_raw_data_y)#shape2111*1

#Step_2
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
#其次画x之间的相关性热力图
# 计算相关系数矩阵
import seaborn as sns
import matplotlib.pyplot as plt
# 设置图片清晰度
plt.rcParams['figure.dpi'] = 600
# 设置 matplotlib 支持中文，使用 SimHei 字体
plt.rcParams['font.family'] = 'SimHei'
# 单独设置英文字体为 Times New Roman
plt.rcParams['font.sans-serif'] = ['SimHei', 'Times New Roman']
# 解决负号显示问题
plt.rcParams['axes.unicode_minus'] = False
# 假设 wind_orignal_data_all_x 已经定义，这里
# 绘制热力图
plt.figure(figsize=(20, 15))

ax = sns.heatmap(obesity_raw_data_x.corr().round(2), annot=True, cmap='coolwarm', vmin=-1, vmax=1,
                 annot_kws={'size': 15})
plt.title('变量间热力图', fontsize=20)
# 设置 x 轴刻度标签（变量名）的字体大小
plt.xticks(fontsize=20)
# 设置 y 轴刻度标签（变量名）的字体大小
plt.yticks(fontsize=20)
# 获取颜色条对象
cbar = ax.collections[0].colorbar
# 调整颜色条刻度标签的字体大小
cbar.ax.tick_params(labelsize=15)
plt.savefig('obesity_raw_data相关性热力图.jpg', dpi=600, bbox_inches='tight',pad_inches=0)
plt.show()

##选择模型和特征筛选方法
"""
1.使用Lasso、户信息法（MIC）、递归特征消除法（RFE）
2.使用随机森林模型进行评判
3.经过实验，最终确定使用rfe作为特征选择的方法，选择了15个特征
"""

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score  # 导入分类评价指标

import warnings
from sklearn.exceptions import DataConversionWarning
# 忽略 DataConversionWarning 警告
warnings.filterwarnings("ignore", category=DataConversionWarning)

# 定义评估函数
# 定义评估函数
def evaluate_model(model, X, y):
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    pre = precision_score(y, y_pred, average='weighted')  # 根据情况选择average参数
    rec = recall_score(y, y_pred, average='weighted')
    f1 = f1_score(y, y_pred, average='weighted')
    return acc, pre, rec, f1

from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_regression, RFE

#2.Lasso 特征选择
obesity_select_lasso = Lasso(alpha=0.01)
obesity_select_lasso.fit(obesity_raw_data_x, obesity_raw_data_y)
selected_features_lasso = obesity_raw_data_x.columns[obesity_select_lasso.coef_ != 0]
obesity_raw_data_x_lasso = obesity_raw_data_x[selected_features_lasso]
#使用随机森林分类模型
obesity_raw_data_x_lasso_rf_model = RandomForestClassifier(random_state=42)
obesity_raw_data_x_lasso_rf_model.fit(obesity_raw_data_x_lasso, obesity_raw_data_y)
acc_rf, pre_rf, rec_rf, f1_rf = evaluate_model(obesity_raw_data_x_lasso_rf_model, obesity_raw_data_x_lasso, obesity_raw_data_y)
print("Lasso 特征选择 + 随机森林回归评估结果：")
print(f"acc: {acc_rf}, pre: {pre_rf}, rec: {rec_rf}, f1: {f1_rf}")
# 输出被选中的特征
print("Lasso 特征筛选中选中的特征有：")
print(selected_features_lasso)
#查看选中特征的数量
num_selected_features = len(selected_features_lasso)
print(f"总共选中了 {num_selected_features} 个特征。")

#3.互信息法（MIC）特征选择
obesity_select_mic = SelectKBest(score_func=mutual_info_regression, k=15)  # 选择前 15 个特征
obesity_raw_data_x_mic = obesity_select_mic.fit_transform(obesity_raw_data_x, obesity_raw_data_y.values.ravel())
#获取被选中特征的布尔掩码
selected_mask_mic = obesity_select_mic.get_support()
#根据布尔掩码获取被选中的特征名称
selected_features_mic = obesity_raw_data_x.columns[selected_mask_mic]
# 打印被选中的特征名称
print("互信息法（MIC）选择的 15 个特征名称：")
print(selected_features_mic)
obesity_raw_data_x_mic_rf_model = RandomForestClassifier(random_state=42)
obesity_raw_data_x_mic_rf_model.fit(obesity_raw_data_x_mic, obesity_raw_data_y)
acc_mic, pre_mic, rec_mic, f1_mic = evaluate_model(obesity_raw_data_x_mic_rf_model, obesity_raw_data_x_mic, obesity_raw_data_y)
print("互信息法（MIC）特征选择评估结果：")
print(f"acc: {acc_rf}, pre: {pre_rf}, rec: {rec_rf}, f1: {f1_rf}")

#4.递归特征消除法（RFE）特征选择
obesity_select_estimator = RandomForestClassifier(random_state=42)
rfe = RFE(obesity_select_estimator, n_features_to_select=15)  # 选择前 15 个特征
obesity_raw_data_x_rfe = rfe.fit_transform(obesity_raw_data_x, obesity_raw_data_y.values.ravel())
# 获取被选中特征的布尔掩码
selected_mask_rfe = rfe.get_support()
# 根据布尔掩码获取被选中的特征名称
selected_features_rfe = obesity_raw_data_x.columns[selected_mask_rfe]
# 打印被选中的特征名称
print("递归特征消除法（RFE）选择的 15 个特征名称：")
print(selected_features_rfe)
# 训练随机森林模型并评估
obesity_raw_data_x_rfe_rf_model = RandomForestClassifier(random_state=42)
obesity_raw_data_x_rfe_rf_model.fit(obesity_raw_data_x_rfe, obesity_raw_data_y)
acc_rfe, pre_rfe, rec_rfe, f1_rfe = evaluate_model(obesity_raw_data_x_rfe_rf_model, obesity_raw_data_x_rfe, obesity_raw_data_y)
print("递归特征消除法（RFE）特征选择评估结果：")
print(f"acc: {acc_rf}, pre: {pre_rf}, rec: {rec_rf}, f1: {f1_rf}")

"""
通过RFE筛选出来的特征一共有15个，具体如下：
['gender', 'age', 'height', 'weight', 'family_history_with_overweight',
       'favc', 'fcvc', 'ncp', 'caec', 'ch2o', 'faf', 'tue', 'calc', 'mtrans',
       'bmi']
下一步就是要把这些特征放到模型里面进行训练，然后挑选出最优模型，最后进行预测，并用shap做可解释分析
"""

##step_2：模型训练
"""
1.需要把预处理+特征工程处理好的数据拿出来形成新的x和y
2.挑选10个主流机器学习模型进行对比
"""
obesity_prepare_data_x = obesity_raw_data_x.drop(['smoke', 'scc'],
                                                 axis=1)
obesity_prepare_data_y = obesity_raw_data_y
print(obesity_prepare_data_x.shape)#shape:（2111，15）
print(obesity_prepare_data_y.shape)#shape:（2111，1）
print(obesity_prepare_data_x.columns)#与筛选出来额报错一致

# 将 obesity_prepare_data_x 和 obesity_prepare_data_y 按列连接
combined_data = pd.concat([obesity_prepare_data_x, obesity_prepare_data_y], axis=1)
# 保存为 CSV 文件
combined_data.to_csv('preprocess_obesity_dataset.csv', index=False)
print("数据已成功保存为 combined_obesity_data.csv 文件。")




