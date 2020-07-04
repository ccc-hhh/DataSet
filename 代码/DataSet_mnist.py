import torchvision
import matplotlib.pyplot as plt


# x_train:train.train_data, y_train:train.train_labels;(60000, 28, 28)(0~10)
# x_test:test.test_data, y_test:test.test_labels;(10000, 28, 28)
def DataSet_mnist():
    train = torchvision.datasets.MNIST(
        root='D:/pycharm/神经网络/数据/mnist/',
        train=True,  # this is train data(60000, 28, 28)
        transform=torchvision.transforms.ToTensor(),  # Converts a PIL.Image or numpy.ndarray to
        # torch.FloatTensor of shape (C x H x W) and normalize in the range [0.0, 1.0]
        download=False,
    )
    test = torchvision.datasets.MNIST(
        root='D:/pycharm/神经网络/数据/mnist/',
        train=False,  # this is test data(10000, 28, 28)
        transform=torchvision.transforms.ToTensor(),  # Converts a PIL.Image or numpy.ndarray to
        # torch.FloatTensor of shape (C x H x W) and normalize in the range [0.0, 1.0]
        download=False,
    )
    return train, test


if __name__ == "__main__":
    my_train, my_test = DataSet_mnist()
    # plot one example
    print(my_train.train_data.size())  # (60000, 28, 28)
    print(my_train.train_labels.size())  # (60000)
    print(my_test.test_data.size())  # (10000, 28, 28)
    print(my_test.test_data.size())  # (10000)
    plt.subplot(2, 1, 1)
    plt.imshow(my_train.train_data[0].numpy(), cmap='gray')
    plt.title('%i' % my_train.train_labels[0])
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.imshow(my_test.test_data[0].numpy(), cmap='gray')
    plt.title('%i' % my_test.test_labels[0])
    plt.legend()
    plt.show()
