import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd


# ['Date', 'Time', 'CO(GT)', 'PT08.S1(CO)', 'NMHC(GT)', 'C6H6(GT)', 'PT08.S2(NMHC)',
#   'NOx(GT)', 'PT08.S3(NOx)', 'NO2(GT)', 'PT08.S4(NO2)', 'PT08.S5(O3)', 'T', 'RH', 'AH']
def DataSet_Random(random_state):
    # 读数据
    AirQuality = pd.read_csv(r"数据/AirQualityUCI.csv", encoding='utf-8')
    # 列名
    # columns = AirQuality.columns.values
    # 全部数据
    data = np.array(AirQuality.values.tolist())
    # print(AirQuality)
    # 时间日期
    date = data[:, 0:2]
    # 空气质量数据
    x = data[:, 2:]
    # 拆分
    x_train, x_test, y_train, y_test = train_test_split(x, date, test_size=0.2, random_state=random_state)
    return  x_train, x_test, y_train, y_test


if __name__ == "__main__":
    x_train, x_test, y_train, y_test = DataSet_Random(0)
    print(y_train)
