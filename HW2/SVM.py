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

# Here are some helper functions
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


# choose learning rate
def small_experiment():
    learning_rate = [0.01, 0.02, 0.05, 0.07, 0.1, 0.2, 0.5]

    # use 1000 example to train, 100 to test
    test_index, train_index = split(train, 100)
    train_index = train_index[:1000]
    small_train = train[train_index]
    small_y = labels[train_index]
    small_test = train[test_index]
    small_target = labels[test_index]

    accuracy = []
    time = []
    # train and record accuracy and time
    for lr in learning_rate:
        start = datetime.datetime.now()
        a = np.random.rand(6)
        b = np.random.rand(1)[0]
        for i in range(5000):
            rand_ind = np.random.choice(range(1000), 1)[0]
            xi = small_train[rand_ind]
            yi = small_y[rand_ind]
            a, b = gradient(a, b, lr, 0.01, xi, yi)
        end = datetime.datetime.now()
        accuracy.append(cal_accuracy(sign(small_test, a, b), small_target))
        time.append((end-start).total_seconds())

    # plot accuracy and time
    plt.scatter(accuracy, time)
    for i, txt in enumerate(learning_rate):
        plt.annotate(txt, (accuracy[i], time[i]))
    plt.show()


# search lambda
lambda_vals = [1e-3, 1e-2, 1e-1, 5e-1, 1]
num_val = 0.1
def search(nda):
    best_lambda = 0
    best_accuracy = 0
    for l in lambda_vals:
        avg_acc = 0
        for round in range(50):
            # train validation split
            val_index, season_index = split(train, int(num_val * len(train)))
            search_train = train[season_index]
            search_yi = labels[season_index]
            validation = train[val_index]
            val_label = labels[val_index]

            # train model
            a = np.random.rand(6)
            b = np.random.rand(1)[0]
            for i in range(5000):
                rand_ind = np.random.choice(range(len(search_train)), 1)[0]
                xi = search_train[rand_ind]
                yi = search_yi[rand_ind]
                a, b = gradient(a, b, nda, l, xi, yi)
            avg_acc += cal_accuracy(sign(validation, a, b), val_label)

        # find the max accuracy
        if avg_acc > best_accuracy:
            best_accuracy = avg_acc
            best_lambda = l
    return best_lambda


# parameters
epochs = 50
held_out = 50
season_steps = 300
acc_steps = 30
m = 1
n = 10


# SVM training
acc_30_steps = []
magnitude = []
def svm(lambda_val):
    a = np.random.rand(6)
    b = np.random.rand(1)[0]
    for i in range(epochs):
        print("Epochs ", i)

        # update nda for each season
        nda = m / (i + n)

        # held out 50 evaluation examples
        eval_index, train_index = split(train, held_out)
        new_train = train[train_index]
        new_yi = labels[train_index]
        new_test = train[eval_index]
        new_label = labels[eval_index]

        for j in range(season_steps):
            # gradient update parameter a and b
            rand_ind = np.random.choice(range(len(new_train)), 1)[0]
            xi = new_train[rand_ind]
            yi = new_yi[rand_ind]
            a, b = gradient(a, b, nda, lambda_val, xi, yi)

            # calculate accuracy every 30 steps
            if (j % acc_steps == 0) & (j > 0):
                predictions = sign(new_test, a, b)
                season_acc = cal_accuracy(predictions, new_label)
                acc_30_steps.append(season_acc)
                magnitude.append(np.linalg.norm(a))
                print("in step ", j, " season accuracy is ", season_acc)

        predictions = sign(new_test, a, b)
        epoch_acc = cal_accuracy(predictions, new_label)
        acc_30_steps.append(epoch_acc)
        magnitude.append(np.linalg.norm(a))
        print("Epochs accuracy is ", epoch_acc)
    return a, b


# get the value of m and n
small_experiment()

# train svm
plot_acc = {}
plot_mag = {}
for l in lambda_vals:
    acc_30_steps = []
    magnitude = []
    a, b = svm(l)
    predictions = sign(test, a, b)
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

# search for best lambda
nda = 0.1
best_l = search(nda)
print("best lambda: ", best_l)
a,b = svm(best_l)
predictions = sign(test, a, b)

# write to submission file
file = open("submission.txt", "w")
print(predictions)
for pred in predictions:
    if pred == -1:
        file.write("<=50K\n")
    else:
        file.write(">50K\n")
file.close()