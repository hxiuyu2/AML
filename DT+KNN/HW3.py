import sys
import collections
from heapq import heappush, heappop


class kdtree():
    def __init__(self, th, attr):
        self.left = None
        self.right = None
        self.parent = None
        self.attribute = attr
        self.threshold = th
        self.points = []


    def build(self, data, index):
        if index > len(data[0] - 1):
            return None
        column = [x[index] for x in data]
        column = sorted(column)
        median = column[int(len(column) / 2)]
        node = kdtree(median, index)
        node.points = data
        node.left = self.build([x for x in data if x[index] <= median], index + 1)
        node.right = self.build([x for x in data if x[index] > median], index + 1)
        node.left.parent = node
        node.right.parent = node
        return node


    def search(self, query, root):
        node = root
        while node is not None:
            temp_attr = node.attribute
            temp_thre = node.threshold
            if query[temp_attr] <= temp_thre:
                node = node.left
            else:
                node = node.right
        return node

    def is_leaf(self):
        if self.left is None and self.right is None:
            return True
        else:
            return False

    def visit(self, target):
        h = []
        near_heap = []
        for cand in self.points:
            distance = euclidean(cand, target[:-1])
            heappush(h, (distance, target))

        heap_size = len(h)
        dist_list = []
        while heap_size > 0:
            dist, lab = heappop(h)
            heap_size -= 1
            if dist in dist_list:
                heappush(near_heap, (dist, lab))
            else:
                if len(near_heap) >= 3:
                    break
                heappush(near_heap, (dist, lab))
                dist_list.append(dist)

        max_dist = dist_list[-1]
        return max_dist, near_heap


def knn_kdtree(train, label, test):
    tree = kdtree()
    dataset = []
    for index, row in enumerate(train):
        line = list(row)
        line.append(label[index])
        dataset.append(line)
    root = tree.build(dataset, 0)

    for point in test:
        dist_heap = []
        node = tree.search(point, root).parent
        dist = 0 # infinite
        while node.parent is not None:
            if point[node.attribute] <= node.threshold:
                if not node.is_leaf():
                    leaf = tree.search(point, node)
                max_dist, near_heap = leaf.visit(point)
                dist, dist_heap = update(max_dist, dist, near_heap, dist_heap)
                if abs(point[node.atttibutes] - node.threshold) <= dist:
                    max_dist, near_heap = node.right.visit(point)
                    dist, dist_heap = update(max_dist, dist, near_heap, dist_heap)
            else:
                max_dist, near_heap = node.right.visit(point)
                dist, dist_heap = update(max_dist, dist, near_heap, dist_heap)
                if abs(point[node.atttibutes] - node.threshold) <= dist:
                    max_dist, near_heap = node.left.visit(point)
                    dist, dist_heap = update(max_dist, dist, near_heap, dist_heap)

            node = node.parent


def update(max_dist, dist, near_heap, dist_heap):
    # update the largest distance
    if max_dist < dist:
        dist = max_dist

    hsize = len(near_heap)
    temp = []
    while hsize > 0:
        d, p = heappop(near_heap)
        heappush(dist_heap, (d, p))

    # update the nearest three points
    d, p = heappop(dist_heap)
    heappush(temp, (d, p))
    d, p = heappop(dist_heap)
    heappush(temp, (d, p))
    d, p = heappop(dist_heap)
    heappush(temp, (d, p))
    dist_heap = temp

    return dist, dist_heap

# read input
def read_inputs(ins):
    X = []
    y = []
    test = []
    for line in ins:
        parts = line.split()
        attr_dict = {}
        label = -1
        for l in parts:
            if label == -1:
                label = int(l)
            else:
                attribute = l.split(":")
                attr_dict[int(attribute[0])] = float(attribute[1])
        od = collections.OrderedDict(sorted(attr_dict.items()))
        row = [v for k, v in od.items()]
        if label != 0:
            X.append(row)
            y.append(label)
        else:
            test.append(row)
    return X, y, test


# def knn(X, y, test):
#     if len(X) < 3:
#         print(min(y))
#         return
#
#     for t in test:
#         h = []
#         for i in range(len(X)):
#             heappush(h, (euclidean(t, X[i]), y[i]))
#
#         heap_size = len(h)
#         label_list = []
#         dist_list = []
#         while heap_size > 0:
#             dist, lab = heappop(h)
#             heap_size -= 1
#             if dist in dist_list:
#                 label_list.append(lab)
#             else:
#                 if len(label_list) >= 3:
#                     break
#                 label_list.append(lab)
#                 dist_list.append(dist)
#         label_list = sorted(label_list)[0:3]
#         # print(label_list)
#         print(majority_vote(label_list))
#
#
def euclidean(a, b):
    diff = [x - y for x, y in zip(a, b)]
    diff = [num ** 2 for num in diff]
    return sum(diff)


def split(X, y):
    total_count = len(y)
    if total_count <= 1:
        return None, y[0]

    # map y: label -> count
    y_set = list(set(y))
    if len(y_set) <= 1:
        return None, y[0]
    y_dict = {}
    for value in y_set:
        y_dict[value] = y.count(value)

    # gini for every attribute
    best_attr = -1
    min_gini = 0  # find max of the sum of sigma(label_count^2)/set_count
    best_threshold = -1
    for col in range(len(X[0])):
        data = [row[col] for row in X]
        value_set = sorted(list(set(data)))

        # this attribute cannot split
        if len(value_set) <= 1:
            continue

        value_index = []

        # every possible threshold
        for split_index in range(len(value_set) - 1):
            value = value_set[split_index]
            value_index.extend([i for i, x in enumerate(data) if x == value])
            set_count = len(value_index)
            label_list = [y[i] for i in value_index]
            label_set = list(set(label_list))
            label_dict = {}
            for l in label_set:
                label_dict[l] = label_list.count(l)

            sum_d1 = 0
            sum_d2 = 0
            for label, count in y_dict.items():
                label_count = 0
                if label in label_dict:
                    label_count = label_dict[label]
                sum_d1 += (label_count) ** 2
                sum_d2 += (y_dict[label] - label_count) ** 2

            gini = sum_d1 / set_count + sum_d2 / (total_count - set_count)
            if gini > min_gini:
                min_gini = gini
                best_attr = col
                best_threshold = (value_set[split_index] + value_set[split_index + 1]) / 2
    return best_threshold, best_attr


def decision_tree(X, y, test):
    threshold, attribute = split(X, y)
    # print("threshold, ", threshold, " ; attribute, ", attribute)

    if threshold is not None:
        set_left = [x for x in X if x[attribute] <= threshold]
        y_left = [y[i] for i, x in enumerate(X) if x[attribute] <= threshold]
        set_right = [x for x in X if x[attribute] > threshold]
        y_right = [y[i] for i, x in enumerate(X) if x[attribute] > threshold]
        l_th, l_attr = split(set_left, y_left)
        r_th, r_attr = split(set_right, y_right)
        # print("l_th, ", l_th, " ; l_attr, ", l_attr)
        # print("r_th, ", r_th, " ; r_attr, ", r_attr)
        
        left_left, left_right, right_left, right_right = -1, -1, -1, -1
        if l_th is not None:
            left_set = []
            right_set = []
            for i, x in enumerate(set_left):
                if x[l_attr] <= l_th:
                    left_set.append(y_left[i])
                else:
                    right_set.append(y_left[i])
            left_left = majority_vote(left_set)
            left_right = majority_vote(right_set)
        if r_th is not None:
            left_set = []
            right_set = []
            for i, x in enumerate(set_right):
                if x[r_attr] <= r_th:
                    left_set.append(y_right[i])
                else:
                    right_set.append(y_right[i])
            right_left = majority_vote(left_set)
            right_right = majority_vote(right_set)

        for data in test:
            if threshold is None:
                print(attribute)
            elif data[attribute] <= threshold:
                if l_th is not None:
                    if data[l_attr] <= l_th:
                        print(left_left)
                    else:
                        print(left_right)
                else:
                    print(l_attr)
            else:
                if r_th is not None:
                    if data[r_attr] <= r_th:
                        print(right_left)
                    else:
                        print(right_right)
                else:
                    print(r_attr)
    else:
        print(attribute)


def majority_vote(set_label):
    label_set = sorted(list(set(set_label)))
    max_count = 0
    max_label = -1
    for l in label_set:
        count = set_label.count(l)
        if count > max_count:
            max_count = count
            max_label = l
    return max_label


inputs = ['1 0:1.5 2:7.1 5:8.1 4:1.1 9:5.1 3:7.8',
          '1 0:7.0 2:1.1 5:5.1 4:0.1 9:2.1 3:1.2',
          '1 0:1.0 2:1.1 5:8.1 4:6.1 9:7.1 3:8.4',
          '2 0:5.0 2:1.1 5:8.1 4:5.1 9:8.1 3:7.2',
          '1 0:1.0 2:1.1 5:8.1 4:6.1 9:7.1 3:8.4',
          '8 0:2.0 2:2.1 5:2.1 4:5.1 9:8.1 3:7.2',
          '8 0:2.0 2:1.1 5:8.1 4:5.1 9:7.1 3:8.4',
          '8 0:5.0 2:1.1 5:5.1 4:5.1 9:7.1 3:7.2',
          '2 0:1.0 2:4.1 5:8.1 4:2.4 9:4.1 3:11.2',
          '2 0:1.0 2:8.1 5:8.1 4:3.1 9:6.1 3:20.2',
          '3 0:1.0 2:1.1 5:4.1 4:2.5 9:8.6 3:4.2',
          '3 0:1.0 2:1.1 5:1.1 4:8.1 9:7.7 3:2.2',
          '0 0:7.0 2:2.2 5:8.1 4:2.2 9:2.5 3:1.2',
          '0 0:4.5 2:5.0 5:2.5 4:7.1 9:6.8 3:10.2',
          '0 0:0.5 2:1.0 5:1.5 4:3.1 9:7.3 3:2.2',
          '0 0:1.5 2:1.0 5:4.5 4:8.1 9:3.6 3:7.2',
          '0 0:7.0 2:7.2 5:8.1 4:2.2 9:2.5 3:1.2',
          '0 0:4.5 2:5.0 5:8.5 4:7.1 9:7.8 3:5.2',
          '0 0:0.5 2:2.0 5:7.5 4:3.1 9:4.3 3:2.2',
          '0 0:1.5 2:1.0 5:4.5 4:8.1 9:3.6 3:2.2']
# filepath = 'germandata.txt'
# inputs = []
# # modified = open('germandata.txt', 'w+')
# with open(filepath) as fp:
#    line = fp.readline()
#    line = line.strip()#.replace(": ", ":") + '\n'
#    # modified.write(line)
#    inputs.append(line)
#    while line:
#        line = fp.readline()
#        if len(line) > 0:
#            line = line.strip()#.replace(": ", ":") + '\n'
#            # modified.write(line)
#            inputs.append(line)
# # modified.close()
train, label, test = read_inputs(inputs) # (sys.stdin)
decision_tree(train, label, test)
print()
knn_kdtree(train, label, test)