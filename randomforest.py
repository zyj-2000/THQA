import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import spearmanr, pearsonr, kendalltau
import numpy as np
from sklearn.model_selection import KFold
from scipy.optimize import curve_fit

def logistic_func(X, bayta1, bayta2, bayta3, bayta4):
    logisticPart = 1 + np.exp(np.negative(np.divide(X - bayta3, np.abs(bayta4))))
    yhat = bayta2 + np.divide(bayta1 - bayta2, logisticPart)
    return yhat

def fit_function(y_label, y_output):
    beta = [np.max(y_label), np.min(y_label), np.mean(y_output), 0.5]
    popt, _ = curve_fit(logistic_func, y_output, y_label, p0=beta, maxfev=100000000)
    y_output_logistic = logistic_func(y_output, *popt)
    return y_output_logistic

# 读取CSV文件
data = pd.read_csv("vvmf_results_a01.csv")

# 提取特征X和目标值y
X = data.iloc[:, 1:25]
y = data.iloc[:, 25]

# 初始化列表用于存储各折的指标
srcc_list = []
plcc_list = []
krcc_list = []
rmse_list = []

# 五折交叉验证
kf = KFold(n_splits=5, shuffle=True, random_state=42)
for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # 使用随机森林回归模型
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42) #100 #42 2 30
    rf_model.fit(X_train, y_train)

    # 进行预测
    rf_predictions = rf_model.predict(X_test)

    # 计算指标
    predict_logistic = fit_function(y_test, rf_predictions)
    test_srcc = spearmanr(y_test, rf_predictions)[0]
    test_plcc = pearsonr(predict_logistic, y_test)[0]
    test_krcc = kendalltau(rf_predictions, y_test)[0]
    test_rmse = np.sqrt(((predict_logistic - y_test) ** 2).mean())

    # 将指标添加到列表
    srcc_list.append(test_srcc)
    plcc_list.append(test_plcc)
    krcc_list.append(test_krcc)
    rmse_list.append(test_rmse)

# 计算平均值
avg_srcc = np.mean(srcc_list)
avg_plcc = np.mean(plcc_list)
avg_krcc = np.mean(krcc_list)
avg_rmse = np.mean(rmse_list)

# 打印结果
print(f"Average SRCC: {avg_srcc:.4f}")
print(f"Average PLCC: {avg_plcc:.4f}")
print(f"Average KRCC: {avg_krcc:.4f}")
print(f"Average RMSE: {avg_rmse:.4f}")
