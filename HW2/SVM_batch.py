import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt

# read file
train = pd.read_csv('train.txt', sep=",", header=None)
train.columns = ['age','workclass','fnlwgt','education','education-num','marital-status','occupation','relationship',
                'race','sex','capital-gain','capital-loss','hours-per-week','native-country','label']
test = pd.read_csv('test.txt', sep=",", header=None)
test.columns = ['age','workclass','fnlwgt','education','education-num','marital-status','occupation','relationship',
                'race','sex','capital-gain','capital-loss','hours-per-week','native-country']


# data preprocess
def normalize(data):
    data = data - np.mean(data, axis=0)
    data = data / np.std(data, axis=0)
    return data


s = train['label']
labels = []
for x in s:
    if x == s[0]:
        labels.append(-1)
    else:
        labels.append(1)
labels = np.array(labels)
CONT_COLS = ['age','fnlwgt','education-num','capital-gain','capital-loss','hours-per-week']
train = train[CONT_COLS].values
test = test[CONT_COLS].values
train = normalize(train)
test = normalize(test)

# ------------------------------------------
# Here are some helper functions
# ------------------------------------------
# calculate accuracy
def cal_accuracy(predictions, yi):
    correct = 0
    for pred, actual in zip(predictions, yi):
        if pred == actual:
            correct += 1
    return correct / len(yi)


# update and gradient
def gradient(a, b, nda, lambda_val, data, yi):
    res = yi * (np.dot(a, data.T) + b)
    if res >= 1:
        temp = (nda * lambda_val) * a
        a -= temp
    else:
        temp = lambda_val * a
        temp -= yi * data
        a -= nda * temp
        b -= nda * (-yi)
    return a, b


# do batch gradient
def batch_gradient(batch_size, new_train, new_yi, a, b, nda, lambda_val):
    batch_a = np.zeros(6)
    batch_b = 0
    rand_ind = np.random.choice(range(len(new_train)), batch)
    for batch_ind in rand_ind:
        # gradient update parameter a and b
        xi = new_train[batch_ind]
        yi = new_yi[batch_ind]
        new_a, new_b = gradient(a, b, nda, lambda_val, xi, yi)
        batch_a += new_a
        batch_b += new_b

    a = (1 / batch) * batch_a
    b = (1 / batch) * batch_b
    return a,b

# predict class
def sign(test_set, a, b):
    result = np.dot(test_set, a.T) + b
    result[result <= 0] = -1
    result[result > 0] = 1
    return result


# train test split
def split(data, size):
    rows = data.shape[0]
    index = np.arange(0, rows)
    np.random.shuffle(index)
    new_index = index[:size]
    rest_index = index[size:]
    return new_index, rest_index


# calculate magnitude
def cal_mag(a):
    ssq = np.sum(a ** 2)
    return ssq


# parameters
epochs = 50
held_out = 50
season_steps = 300
acc_steps = 30
m = 1
n = 10
batch = 10


# SVM training
acc_30_steps = []
magnitude = []
def svm(lambda_val, train, labels):
    a = np.random.rand(6)
    b = np.random.rand(1)[0]
    count = 0
    stable_acc = 0
    for i in range(epochs):
        # held out 50 evaluation examples
        eval_index, train_index = split(train, held_out)
        new_train = train[train_index]
        new_yi = labels[train_index]
        new_test = train[eval_index]
        new_label = labels[eval_index]

        for j in range(season_steps):
            # update nda for each season
            nda = batch*m / ((i+1)*j + batch*n)
            a, b = batch_gradient(batch, new_train, new_yi, a, b, nda, lambda_val)
            predictions = sign(new_test, a, b)
            season_acc = cal_accuracy(predictions, new_label)
            if stable_acc != season_acc:
                stable_acc = season_acc
                count = 0
            else:
                count += 1

            # calculate accuracy every 30 steps
            if (j % acc_steps == 0) & (j > 0):
                acc_30_steps.append(season_acc)
                magnitude.append(cal_mag(a))

        predictions = sign(new_test, a, b)
        epoch_acc = cal_accuracy(predictions, new_label)
        acc_30_steps.append(epoch_acc)
        magnitude.append(cal_mag(a))
        print("Epochs accuracy is ", epoch_acc)
    return a, b


# ------------------------------------------
# Here is the start point of overall process
# ------------------------------------------
lambda_vals = [1e-3, 1e-2, 1e-1, 5e-1, 1]
num_val = 0.1

# train validation split
val_index, season_index = split(train, int(num_val * len(train)))
search_train = train[season_index]
search_yi = labels[season_index]
validation = train[val_index]
val_label = labels[val_index]

# train svm
plot_acc = {}
plot_mag = {}
best_l = 0
best_acc = 0
best_a = 0
best_b = 0
for l in lambda_vals:
    acc_30_steps = []
    magnitude = []
    a, b = svm(l, search_train, search_yi)
    predictions = sign(validation, a, b)
    cur_acc = cal_accuracy(predictions, val_label)
    if cur_acc > best_acc:
        best_acc = cur_acc
        best_l = l
        best_a = a
        best_b = b
    print("for lambda = ", l, " the accuracy on validation set is ", cur_acc)
    plot_acc["reg="+str(l)] = acc_30_steps
    plot_mag["reg="+str(l)] = magnitude
plot_mag['x'] = range(len(acc_30_steps))
plot_acc['x'] = range(len(magnitude))

# multiple line plot
df = pd.DataFrame(plot_acc)
axes = plt.gca()
axes.set_ylim([0, 1])
plt.xlabel("Steps")
plt.ylabel("Accuracy")
for l in lambda_vals:
    plt.plot('x', "reg="+str(l), data=df)
plt.legend()
plt.savefig('accuracy.png')

df = pd.DataFrame(plot_mag)
plt.xlabel("Steps")
plt.ylabel("Magnitude")
for l in lambda_vals:
    plt.plot('x', "reg="+str(l), data=df)
plt.legend(loc='upper right')
plt.savefig('magnitude.png')

print("best lambda: ", best_l)
predictions = sign(test, best_a, best_b)

# write to submission file
file = open("submission.txt", "w")
for pred in predictions:
    if pred == -1:
        file.write("<=50K\n")
    else:
        file.write(">50K\n")
file.close()