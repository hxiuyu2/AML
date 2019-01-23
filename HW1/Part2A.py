from mnist import MNIST
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt


mndata = MNIST('./data')
images, labels = mndata.load_training()
test_img, test_y = mndata.load_testing()

# Calculate P(y)
labels = pd.Series(labels)
p_y = pd.DataFrame(labels.value_counts() / len(labels))
p_y = p_y.sort_index()

train_y = np.array(labels)
test_y = np.array(test_y)

# Record mean for all pixel
mean_pixel = []

def threshhold(a):
    if a > 255/2:
        return 1
    else:
        return 0

def stretch(data):
    data = pd.DataFrame(data)
    data = data[(data.T != 0).any()]
    data = data.loc[:, (data != 0).any(axis=0)]
    # print(data.shape)
    img = data.values
    newimg = cv2.resize(img, (20, 20))
    array = np.array(newimg)
    return array

def gaussain(train, test):
    # Bayes Calculation
    all_train = np.empty(shape=(0, 60000))
    all_test = np.empty(shape=(0, 10000))
    for class_num in range(10):
        index = np.where(train_y == class_num)
        slice = train[index]
        mean = np.mean(slice, axis=0)
        var = np.var(slice, axis=0)
        mean_pixel.append(mean.flatten())
        temp = []
        for row in train:
            sum = np.log(p_y[0][class_num])
            for r, m, v in zip(row, mean, var):
                if v == 0:
                    continue
                else:
                    sum += (-0.5) * (r - m) ** 2 / v - 0.5 * np.log(2 * np.pi * v)
            temp.append(sum)
        p_train = np.array(temp)
        all_train = np.vstack((all_train, p_train))

        temp = []
        for row in test:
            sum = np.log(p_y[0][class_num])
            for r, m, v in zip(row, mean, var):
                if v == 0:
                    continue
                else:
                    sum += (-0.5) * (r - m) ** 2 / v - 0.5 * np.log(2 * np.pi * v)
            temp.append(sum)
        p_test = np.array(temp)
        all_test = np.vstack((all_test, p_test))
    all_train = all_train.T
    all_test = all_test.T
    return all_train, all_test

def bernoulli(train, test):
    # Bernoulli Calculation
    all_train = np.empty(shape=(0, 60000))
    all_test = np.empty(shape=(0, 10000))
    for class_num in range(10):
        index = np.where(train_y == class_num)
        slice = train[index]
        mean = np.mean(slice, axis=0)
        mean_pixel.append(mean.flatten())
        temp = []
        for row in train:
            sum = np.log(p_y[0][class_num])
            for r, m in zip(row, mean):
                if m == 0 or m == 1:
                    continue
                else:
                    # print("m = ", m)
                    sum += r * np.log(m) + (1-r) * np.log(1-m)
            temp.append(sum)
        p_train = np.array(temp)
        all_train = np.vstack((all_train, p_train.T))

        temp = []
        for row in test:
            sum = np.log(p_y[0][class_num])
            for r, m in zip(row, mean):
                if m == 0 or m == 1:
                    continue
                else:
                    sum += r * np.log(m) + (1-r) * np.log(1-m)
            temp.append(sum)
        p_test = np.array(temp)
        all_test = np.vstack((all_test, p_test.T))
    all_train = all_train.T
    all_test = all_test.T
    return all_train, all_test

def preprocess(images, need_stretch):
    data = []
    for img in images:
        newimg = np.array(img, dtype='uint8')
        newimg = newimg.reshape((28, 28))
        vfunc = np.vectorize(threshhold)
        newimg = vfunc(newimg)
        newimg = np.array(newimg, dtype='uint8')
        if need_stretch:
            newimg = stretch(newimg)
        newimg = newimg.flatten()
        data.append(newimg)
    data = np.array(data)
    return data

def cal_accuracy(predictions, target):
    # Making predictions
    preds = []
    for train_res in predictions:
        preds.append(np.argmax(train_res))

    # Calculate accuracy
    correct = 0
    for pred, actual in zip(preds, target):
        if pred == actual:
            correct += 1
    return correct / len(preds)

print("Gaussain + untouched")
train = preprocess(images, False)
test = preprocess(test_img, False)
all_train, all_test = gaussain(train, test)
print("Training Set Accuracy: ",cal_accuracy(all_train, train_y))
print("Test Set Accuracy: ",cal_accuracy(all_test, test_y))

plt.figure(figsize=(9,9))
for i in range(10):
    plt.subplot(3,4, i+1)
    img = np.array(mean_pixel[i]).reshape(28,28)
    plt.imshow(img)
plt.show()

print("Gaussain + stretched")
train = preprocess(images, True)
test = preprocess(test_img, True)
all_train, all_test = gaussain(train, test)
print("Training Set Accuracy: ",cal_accuracy(all_train, train_y))
print("Test Set Accuracy: ",cal_accuracy(all_test, test_y))

print("Bernoulli + untouched")
mean_pixel = []
train = preprocess(images, False)
test = preprocess(test_img, False)
all_train, all_test = bernoulli(train, test)
print("Training Set Accuracy: ",cal_accuracy(all_train, train_y))
print("Test Set Accuracy: ",cal_accuracy(all_test, test_y))

print("Bernoulli + stretched")
train = preprocess(images, True)
test = preprocess(test_img, True)
all_train, all_test = bernoulli(train, test)
print("Training Set Accuracy: ",cal_accuracy(all_train, train_y))
print("Test Set Accuracy: ",cal_accuracy(all_test, test_y))