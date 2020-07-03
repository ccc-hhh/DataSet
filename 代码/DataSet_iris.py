from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd


# class：[setosa, versicolor, virginica]
# feature：[sepallength, sepalwidth, petallength, petalwidth]
def iris_DataSet(random_state):
    iris_data = load_iris()
    feature = np.array(iris_data['data'])
    target = np.array(iris_data['target'])
    x_train, x_test, y_train, y_test = train_test_split(feature, target, test_size=0.2, random_state=random_state)
    return x_train, x_test, y_train, y_test


def iris():
    iris_data = load_iris()
    print(iris_data)
    iris = pd.DataFrame(iris_data['data'], columns=iris_data['feature_names'])
    iris = pd.merge(iris, pd.DataFrame(iris_data['target'], columns=['species']), left_index=True, right_index=True)
    labels = dict(zip([0, 1, 2], iris_data['target_names']))
    iris['species'] = iris['species'].apply(lambda x: labels[x])
    return iris


# 对数据进行归一化处理
def normalized(x_data):
    e = 1e-7  # 防止出现0
    for i in range(x_data.shape[1]):
        max_num = np.max(x_data[:, i])
        min_num = np.min(x_data[:, i])
        x_data[:, i] = (x_data[:, i] - min_num + e) / (max_num - min_num + e)
    return x_data


if __name__ == "__main__":
    x_train, x_test, y_train, y_test = iris_DataSet(random_state=0)
    print(normalized(x_train))
    print(y_train)
    print(iris())
