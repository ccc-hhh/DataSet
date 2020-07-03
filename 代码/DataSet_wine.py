import numpy as np
from sklearn.model_selection import train_test_split


def DataSet_Random(random_state):
    # 读数据
    wine = np.genfromtxt("数据/wine.data", delimiter=",")
    X = wine[:, 1:]
    # 标签第一列
    y = wine[:, 0]
    y[y == 1] = 0
    y[y == 2] = 1
    y[y == 3] = 2
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
    return x_train, x_test, y_train, y_test


# 对数据进行归一化处理
def normalized(x_data):
    e = 1e-7  # 防止出现0
    for i in range(x_data.shape[1]):
        max_num = np.max(x_data[:, i])
        min_num = np.min(x_data[:, i])
        x_data[:, i] = (x_data[:, i] - min_num + e) / (max_num - min_num + e)
    return x_data


if __name__ == "__main__":
    x_train, x_test, y_train, y_test = DataSet_Random(0)
    print(y_train)
    print(x_train.shape)
