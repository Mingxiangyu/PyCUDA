# -*- codeing = utf-8 -*-
# @Time :2023/1/16 16:20
# @Author :xming
# @Version :1.0
# @Descriptioon :
# @Link : https://mp.weixin.qq.com/s?__biz=MzI3MzM0ODU4Mg==&mid=2247498405&idx=1&sn=acc2f0534f23bc494b502cfaed76aae0&chksm=eb26182cdc51913a9a26e780537b1d213e613d56e22b5ecf219f39e800f4bfa6647610b27c06&scene=126&sessionid=1673853752&subscene=227&key=5835a3957d608625272ed1ebeb8459f59c8e8406cea3e7256deda4d0b40c3b2ca3fe79c8548e0839f782be9d30bc37a1cc6e3f333de35887d7aab17150b2b3d2739e41380c603cbd3a41848bb9b369bfd8a09dda5837a0d74bc8641b87eacdcddf91c43d596457bd0dfce422288c05ecd47e19d58d91fd1dd4057eadadeed11b&ascene=7&uin=MjQzMzMyOTUwMA%3D%3D&devicetype=Windows+11+x64&version=63080029&lang=zh_CN&exportkey=n_ChQIAhIQgEZH4TnMwcwmGiy5duYX9xLYAQIE97dBBAEAAAAAAIhvBwuWMTsAAAAOpnltbLcz9gKNyK89dVj0OwSM%2BWhLtIUrIVC6jBbTmgNDlpt5JJVJjj8NqiOkC3p4uSSTKLTNRimKImyWNjY72AMeyJj4%2FVzYtCOdaCPR7nLIPBkxH33KO%2F2DHXYfslPmHoccqG0itiaJuB6Q9JCUDNH0OfDsnhrkORUYPXOZrTzrkGvqI%2BRClqSTZTp8%2BwdwSpG0d8xhmB0lQGe8iR1KwJP595kW2tR3FA13o1tRwZW55z0wUqhdoLlpvDmFBZ6LYA%3D%3D&acctmode=0&pass_ticket=DP18jJOd2bShFCqEa9jXOH2iPN3UiGWSTI7RC8veq0hllLwg3R%2BWPuFX5htO3%2Bu1i21SQT31Glr6zvmXZtMB5Q%3D%3D&wx_header=1&fontgear=2
# @File :  使用随机森林模型预测机票价格.py
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_style('whitegrid')

flights = pd.read_excel('./Data_Train.xlsx', engine='openpyxl')
print(flights.head())

# 看看所有字段的基本信息：
flights.info()

# 其他非零值数量均为10683，只有路线和停靠站点数是10682，删除缺失数据行，让非零值达到一致数量
flights.dropna(inplace=True)
flights.info()

# 接下来看看航空公司的分布特征：
sns.countplot('Airline', data=flights)
plt.xticks(rotation=90)
# plt.show()

# 看看始发地的分布
sns.countplot('Source', data=flights)
plt.xticks(rotation=90)
# plt.show()

# 停靠站点的数量分布
sns.countplot('Total_Stops', data=flights)
plt.xticks(rotation=90)
# plt.show()

# 有多少数据含有额外信息
plot = plt.figure()
sns.countplot('Additional_Info', data=flights)
plt.xticks(rotation=90)
# plt.show()

# 首先转换时间格式
flights['Date_of_Journey'] = pd.to_datetime(flights['Date_of_Journey'])
flights['Dep_Time'] = pd.to_datetime(flights['Dep_Time'], format='%H:%M').dt.time
flights['Duration'] = flights['Duration'].str.replace('h', '*60').str.replace(' ', '+').str.replace('m', '*1').apply(
    eval)

# 研究一下出发时间和价格的关系
flights['weekday'] = flights[['Date_of_Journey']].apply(lambda x: x.dt.day_name())
sns.barplot('weekday', 'Price', data=flights)
# plt.show()

# 月份和机票价格的关系
flights["month"] = flights['Date_of_Journey'].map(lambda x: x.month_name())
sns.barplot('month', 'Price', data=flights)
# plt.show()

# 起飞时间和价格的关系
flights['Dep_Time'] = flights['Dep_Time'].apply(lambda x: x.hour)
flights['Dep_Time'] = pd.to_numeric(flights['Dep_Time'])
sns.barplot('Dep_Time', 'Price', data=flights)
# plot.show()

# 把那些和价格没有关联关系的字段直接去除掉.
flights.drop(['Route', 'Arrival_Time', 'Date_of_Journey'], axis=1, inplace=True)
print(flights.head())
flights.info()

"""
模型训练
"""
# 将字符串变量使用数字替代
from sklearn.preprocessing import LabelEncoder

var_mod = ['Airline', 'Source', 'Destination', 'Additional_Info', 'Total_Stops', 'weekday', 'month', 'Dep_Time']
le = LabelEncoder()
for i in var_mod:
    flights[i] = le.fit_transform(flights[i])
print(flights.head())

# 对每列数据进行特征缩放，提取自变量（x）和因变量（y）
flights.corr()


def outlier(df):
    for i in df.describe().columns:
        Q1 = df.describe().at['25%', i]
        Q3 = df.describe().at['75%', i]
        IQR = Q3 - Q1
        LE = Q1 - 1.5 * IQR
        UE = Q3 + 1.5 * IQR
        df[i] = df[i].mask(df[i] < LE, LE)
        df[i] = df[i].mask(df[i] > UE, UE)
    return df


flights = outlier(flights)
x = flights.drop('Price', axis=1)
y = flights['Price']

# 划分测试集和训练集：
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=101)

# 使用随机森林进行模型训练：
from sklearn.ensemble import RandomForestRegressor

rfr = RandomForestRegressor(n_estimators=100)
rfr.fit(x_train, y_train)

# 在随机森林中，我们有一种根据数据的相关性来确定特征重要性的方法：
features = x.columns
importances = rfr.feature_importances_
indices = pd.np.argsort(importances)
plt.figure(1)
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), features[indices])
plt.xlabel('Relative Importance')
plt.show()

# 对划分的测试集进行预测，得到结果：
predictions = rfr.predict(x_test)
plt.scatter(y_test, predictions)
plt.show()

"""
模型评价

这样看不直观，接下来我们要数字化地评价这个模型。
"""
from sklearn import metrics

print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', pd.np.sqrt(metrics.mean_squared_error(y_test, predictions)))
# r2越接近1说明模型效果越好，这个模型的分数是0.74，算是很不错的模型了。
print('r2_score:', (metrics.r2_score(y_test, predictions)))

# 看看其残差直方图是否符合正态分布
sns.distplot((y_test-predictions),bins=50)
plt.show()