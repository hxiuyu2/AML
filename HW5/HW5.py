from os import listdir
from os.path import isfile, join
from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.ensemble import RandomForestClassifier


activities = ['Brush_teeth', 'Climb_stairs', 'Comb_hair', 'Descend_stairs', 'Drink_glass', 'Eat_meat',
              'Eat_soup', 'Getup_bed', 'Liedown_bed', 'Pour_water', 'Sitdown_chair',
              'Standup_chair', 'Use_telephone', 'Walk']
root_path = 'HMP_Dataset/'

window_size = 32
k_value = 480
overlap = 24


class ActivityData:
    def __init__(self, chunks, filename, act_label):
        self.data = chunks
        self.file = filename
        self.label = act_label


def read_data(chunk_size, shift):
    all_data = []
    all_activity = []
    file_count = {}
    for activity in activities:
        path = root_path + activity + '/'
        fcnt = 0
        for f in listdir(path):
            if 'Accelerometer' not in f:
                continue
            filename = join(path, f)
            fcnt += 1
            if isfile(filename):
                file = open(filename, 'r')
                pile_up_lines = []
                one_file = []
                for line in file:
                    result = line.rstrip().split()
                    result = list(map(int, result))
                    pile_up_lines.extend(result)

                start_index = range(0, len(pile_up_lines), shift)
                for i in start_index:
                    if i+chunk_size >= len(pile_up_lines):
                        break
                    chunk = pile_up_lines[i: i+chunk_size]
                    all_data.append(chunk)
                    one_file.append(chunk)
                file_data = ActivityData(one_file, f, activity)
                all_activity.append(file_data)
        file_count[activity] = fcnt
    return all_data, all_activity, file_count


def train(data, i, j, k):
    obj_train = data[i] + data[j]
    obj_test = data[k]
    X_train = [obj.data for obj in obj_train]
    X_test = [obj.data for obj in obj_test]
    y_train = [obj.label for obj in obj_train]
    y_test = [obj.label for obj in obj_test]
    clf = RandomForestClassifier(n_estimators=100, max_depth=7)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    count = 0
    for val, pred in zip(y_test, predictions):
        if val == pred:
            count += 1
    acc = count / len(predictions)
    print('fold accuracy: ', acc)
    return acc


def draw_hist(hist_data):
    for act_name in activities:
        plt.bar(range(k_value), hist_data[act_name])
        plt.title(act_name)
        plt.savefig('image/' + act_name +'_' + str(k_value) + '.png')
        plt.clf()


# use standard K-means to fit and predict
def vectorize(cluster_num, chunk_size, shift):
    all_data, all_activity, file_count = read_data(chunk_size, shift)
    cluster = KMeans(n_clusters=cluster_num)
    km_model = cluster.fit(all_data)
    hist_data = {}
    vectorized_data = {}
    for one_activity in all_activity:
        features = pd.Series(km_model.predict(one_activity.data))
        features = features.value_counts()
        counts = pd.Series(features, index=range(cluster_num))
        counts = counts.fillna(0)
        if one_activity.label not in vectorized_data:
            vectorized_data[one_activity.label] = []
        vectorized_data[one_activity.label].append(ActivityData(counts, one_activity.file, one_activity.label))

        counts = counts / file_count[one_activity.label]
        if one_activity.label in hist_data:
            df = hist_data[one_activity.label]
            counts = counts.add(df)
        hist_data[one_activity.label] = counts
    return vectorized_data, hist_data


kfold_data = [[], [], []]
vectorized_data, hist_data = vectorize(k_value, window_size, overlap)
draw_hist(hist_data)
for act_name in activities:
    label_data = vectorized_data[act_name]
    numbers = math.ceil(len(label_data) / 3)
    kfold_data[0].extend(label_data[: int(numbers)])
    kfold_data[1].extend(label_data[int(numbers): int(numbers*2)])
    kfold_data[2].extend(label_data[int(numbers*2):])

acc = 0
max_acc = 0
max_round = 0
cur_acc = train(kfold_data, 0, 1, 2)
acc += cur_acc / 3
if cur_acc > max_acc:
    max_round = 0
    max_acc = cur_acc
cur_acc = train(kfold_data, 1, 2, 0)
acc += cur_acc / 3
if cur_acc > max_acc:
    max_round = 1
    max_acc = cur_acc
cur_acc = train(kfold_data, 0, 2, 1)
acc += cur_acc / 3
if cur_acc > max_acc:
    max_round = 2
    max_acc = cur_acc
print('average accuracy', acc)
print('max accuracy round', max_round)