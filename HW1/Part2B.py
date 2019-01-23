from mnist import MNIST
import numpy as np
import pandas as pd
import cv2
from sklearn.ensemble import RandomForestClassifier as rfc

mndata = MNIST('./data')
images, labels = mndata.load_training()
test_img, test_y = mndata.load_testing()
train_y = np.array(labels)
test_y = np.array(test_y)

def threshhold(data):
    data = pd.DataFrame(data)
    data = data.applymap(lambda x: 1 if x > 255/2 else 0)
    return data

def stretch(data):
    data = pd.DataFrame(data)
    data = data[(data.T != 0).any()]
    data = data.loc[:, (data != 0).any(axis=0)]
    # print(data.shape)
    img = data.values
    newimg = cv2.resize(img, (20, 20))
    array = np.array(newimg)
    return array

def preprocess(images, need_stretch):
    data = []
    for img in images:
        newimg = np.array(img, dtype='uint8')
        newimg = newimg.reshape((28, 28))
        newimg = threshhold(newimg).values
        newimg = np.array(newimg, dtype='uint8')
        if need_stretch:
            newimg = stretch(newimg)
        newimg = newimg.flatten()
        data.append(newimg)
    data = np.array(data)
    # print(data.shape)
    return data

def cal_accuracy(predictions, target):
    correct = 0
    for pred, actual in zip(predictions, target):
        if pred == actual:
            correct += 1
    return correct / len(predictions)

def decision_tree(num_tree, depth, need_stretch):
    train = preprocess(images, need_stretch)
    test = preprocess(test_img, need_stretch)
    model = rfc(n_estimators=num_tree, max_depth=depth)
    model.fit(train, train_y)
    train_res = model.predict(train)
    test_res = model.predict(test)
    return train_res, test_res

print("10 trees + 4 depth + untouched")
train_res, test_res = decision_tree(10, 4, False)
print("Train set accuracy: ", cal_accuracy(train_res, train_y))
print("Test set accuracy: ", cal_accuracy(test_res, test_y))

print("10 trees + 4 depth + stretched")
train_res, test_res = decision_tree(10, 4, True)
print("Train set accuracy: ", cal_accuracy(train_res, train_y))
print("Test set accuracy: ", cal_accuracy(test_res, test_y))

print("10 trees + 16 depth + untouched")
train_res, test_res = decision_tree(10, 16, False)
print("Train set accuracy: ", cal_accuracy(train_res, train_y))
print("Test set accuracy: ", cal_accuracy(test_res, test_y))

print("10 trees + 16 depth + stretched")
train_res, test_res = decision_tree(10, 16, True)
print("Train set accuracy: ", cal_accuracy(train_res, train_y))
print("Test set accuracy: ", cal_accuracy(test_res, test_y))

print("30 trees + 4 depth + untouched")
train_res, test_res = decision_tree(30, 4, False)
print("Train set accuracy: ", cal_accuracy(train_res, train_y))
print("Test set accuracy: ", cal_accuracy(test_res, test_y))

print("30 trees + 4 depth + stretched")
train_res, test_res = decision_tree(30, 4, True)
print("Train set accuracy: ", cal_accuracy(train_res, train_y))
print("Test set accuracy: ", cal_accuracy(test_res, test_y))

print("30 trees + 16 depth + untouched")
train_res, test_res = decision_tree(30, 16, False)
print("Train set accuracy: ", cal_accuracy(train_res, train_y))
print("Test set accuracy: ", cal_accuracy(test_res, test_y))

print("30 trees + 16 depth + stretched")
train_res, test_res = decision_tree(30, 16, True)
print("Train set accuracy: ", cal_accuracy(train_res, train_y))
print("Test set accuracy: ", cal_accuracy(test_res, test_y))