import numpy as np
import pandas as pd

data = pd.read_csv('pima-indians-diabetes.csv')

accuracy = []
for i in range(15):
    # train test split
    rows = data.shape[0]
    index = np.arange(0, rows)
    np.random.shuffle(index)
    test_index = index[:int(0.2 * rows)]
    train_index = index[int(0.2 * rows):]
    train = data.iloc[train_index]
    test = data.iloc[test_index]

    # Bayes Calculation
    num_0 = data[data['1'] == 0].shape[0]
    p0 = num_0 / rows
    p1 = 1 - p0
    px0 = np.zeros(len(test)) + np.log(p0)
    px1 = np.zeros(len(test)) + np.log(p1)
    for col in train.columns: # ['6', '148', '0', '0.627']: #
        if col == '1':
            continue
        mean0 = np.mean(train.loc[train['1'] == 0, col])
        var0 = np.var(train.loc[train['1'] == 0, col])
        mean1 = np.mean(train.loc[train['1'] == 1, col])
        var1 = np.var(train.loc[train['1'] == 1, col])
        px0 += (-0.5) * (test[col].values - mean0) ** 2 / var0 - 0.5 * np.log(2*np.pi*var0)
        px1 += (-0.5) * (test[col].values - mean1) ** 2 / var1 - 0.5 * np.log(2*np.pi*var1)

    # for col in ['72', '35', '33.6', '50']:
    #     mean0 = np.mean(train.loc[(train['1'] == 0) & (train[col] != 0), col])
    #     var0 = np.var(train.loc[(train['1'] == 0) & (train[col] != 0), col])
    #     mean1 = np.mean(train.loc[(train['1'] == 1) & (train[col] != 0), col])
    #     var1 = np.var(train.loc[(train['1'] == 1) & (train[col] != 0), col])
    #     px0 += (-0.5) * (test[col].values - mean0) ** 2 / var0 - 0.5 * np.log(2*np.pi*var0)
    #     px1 += (-0.5) * (test[col].values - mean1) ** 2 / var1 - 0.5 * np.log(2*np.pi*var1)

    # Making predictions
    predictions = []
    for c0, c1 in zip(px0, px1):
        if c0 > c1:
            predictions.append(0)
        else:
            predictions.append(1)

    # Calculate accuracy
    correct = 0
    for pred, actual in zip(predictions, test['1']):
        if pred == actual:
            correct+=1
    accuracy.append(correct / len(predictions))

# Average accuracy
sum = 0
for acc in accuracy:
    sum += acc
avg = sum / len(accuracy)
print("Unprocessed data accuracy: ", avg)
# print("0 value elements ignored: ", avg)


# def bounding(data):
#     data = pd.DataFrame(data)
#     df = data.sum(axis=1)
#     x_center = int((df.nonzero()[0][0] + df.nonzero()[0][-1]) / 2)
#     df = data.sum(axis=0)
#     y_center = int((df.nonzero()[0][0] + df.nonzero()[0][-1]) / 2)
#     data = data.iloc[x_center-9:x_center+11, y_center-9:y_center+11]
#     print(data.shape)
#     return data
#