# sklearn 数据集
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


def dataset_demo():
    iris = load_iris()

    # load dataset
    print(iris.data)
    # print(iris["DESCR"])
    # 每个特征列的特征名称
    print(iris.feature_names)
    print(iris.data.shape)
    # 标签值，即每个样本行的类别，此处分为三类0,1,2。
    print(iris.target)

    # split dataset
    # x 数据集特征值；y 数据集标签值
    # 数据，目标值，测试集比例，随机数种子（随机划分因子，固定值则每次划分的数据是一致的）
    x_test, x_train, y_test, y_train = train_test_split(iris.data, iris.target, test_size=0.25, random_state=43)

    print("特征值: ", x_test, x_train)
    print("标签值: ", y_test, y_train)


if __name__ == '__main__':
    dataset_demo()