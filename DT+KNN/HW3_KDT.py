import sys
import collections
from heapq import heappush, heappop


# read input
def read_inputs(ins):
    X = []
    y = []
    test = []
    data = []
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
            line = list(row)
            line.append(label)
            data.append(line)
        else:
            test.append(row)
    return X, y, test, data


class KDNode():
    def __init__(self):
        self.left = None
        self.right = None
        self.parent = None
        self.split = []
        self.attribute = 0
        self.threshold = 0


def build_tree(data):
    if len(data) == 0:
        return None

    if len(data) == 1:
        node = KDNode()
        node.split = data[0]
        return node

    max_var = 0
    max_var_ind = 0
    row_count = len(data)

    # find max variance of attributes
    for index in range(len(data[0]) - 1):
        column = [x[index] for x in data]
        avg = sum(column) / row_count
        diff = [(x - avg)**2 for x in column]
        variance = sum(diff)
        if variance > max_var:
            max_var = variance
            max_var_ind = index

    # find split point
    sorted_points = sorted(data, key=lambda x: x[max_var_ind])
    split_point = sorted_points[int(row_count / 2)]

    # recursive on left and right
    root = KDNode()
    root.split = split_point
    root.attribute = max_var_ind
    root.threshold = sorted_points[int(row_count / 2)][max_var_ind]
    root.left = build_tree(sorted_points[:int(row_count / 2)])
    if root.left is not None:
        root.left.parent = root
    root.right = build_tree(sorted_points[int(row_count / 2) + 1:])
    if root.right is not None:
        root.right.parent = root
    return root


def walk_tree(root, query):
    # path = []
    node = root
    ret = None
    while node is not None:
        # path.append(node)
        ret = node
        if query[node.attribute] <= node.threshold:
            node = node.left
        else:
            node = node.right
    return ret # path[:-1]


def euclidean(a, b):
    diff = [x - y for x, y in zip(a, b)]
    diff = [num ** 2 for num in diff]
    return sum(diff)


def visit(node, query, max_dist, dist_heap, visited):
    # if query[node.attribute] <= node.threshold:
    #     if node.left is not None:
    #         max_dist, dist_heap = update(node.left, query, max_dist, dist_heap)
    # else:
    #     if node.right is not None:
    #         max_dist, dist_heap = update(node.right, query, max_dist, dist_heap)
    #
    # # see if overlap with the other subtree
    # if len(dist_heap) < 3 or (query[node.attribute] - node.threshold)**2 < max_dist:
    #     # visit the other half
    #     if query[node.attribute] <= node.threshold:
    #         if node.right is not None:
    #             max_dist, dist_heap = update(node.right, query, max_dist, dist_heap)
    #             max_dist, dist_heap = visit(node.right, query, max_dist, dist_heap)
    #     else:
    #         if node.left is not None:
    #             max_dist, dist_heap = update(node.left, query, max_dist, dist_heap)
    #             max_dist, dist_heap = visit(node.left, query, max_dist, dist_heap)
    if node in visited or node is None:
        return max_dist, dist_heap

    max_dist, dist_heap = update(node, query, max_dist, dist_heap)
    visited.add(node)
    parent = node.parent
    if parent is None:
        return max_dist, dist_heap

    if len(dist_heap) < 3 or (query[parent.attribute] - parent.threshold)**2 < max_dist:
        if node is parent.left:
            half = walk_tree(parent.right, query)
        else:
            half = walk_tree(parent.left, query)
        max_dist, dist_heap = visit(half, query, max_dist, dist_heap, visited)
    return visit(parent, query, max_dist, dist_heap, visited)
    # return max_dist, dist_heap


def update(node, query, max_dist, dist_heap):
    if node is None:
        return max_dist, dist_heap

    # calculate euclidean between query and split
    distance = euclidean(node.split, query)

    if len(dist_heap) < 3:
        heappush(dist_heap, (-distance, node.split))
        max_dist = max(max_dist, distance)
        return max_dist, dist_heap

    # update distance and near node set(heap)
    if max_dist < distance:
        return max_dist, dist_heap
    elif max_dist > distance:
        heappush(dist_heap, (-distance, node.split))
        if len(dist_heap) > 3:
            heappop(dist_heap)
            max_dist = -dist_heap[0][0]
    else:
        # if two points have the same distance, keep the smaller label
        temp = []
        heappush(temp, (node.split[-1], node.split))
        while len(dist_heap) > 0:
            _, max_p = dist_heap[0]
            if max_dist == distance:
                _, max_p = heappop(dist_heap)
                heappush(temp, (max_p[-1], max_p))
            else:
                break

        while len(dist_heap) < 3:
            _, point = heappop(temp)
            heappush(dist_heap,(distance, point))

    return max_dist, dist_heap


def knn_kdtree(data, test):
    # training
    root = build_tree(data)

    # predict
    for point in test:
        path = walk_tree(root, point)
        dist_heap = []
        visited = set()
        max_dist = 0
        max_dist, dist_heap = visit(path, point, max_dist, dist_heap, visited)
        # for node in reversed(path):
        #     max_dist, dist_heap = visit(node, point, max_dist, dist_heap)

        # form label list
        labels = []
        # print(dist_heap)
        while len(dist_heap) > 0:
            _, point_lab = heappop(dist_heap)
            # print(distance)
            labels.append(point_lab[-1])
        print(majority_vote(labels))


def knn(X, y, test):
    if len(X) < 3:
        print(min(y))
        return

    for t in test:
        h = []
        for i in range(len(X)):
            heappush(h, (euclidean(t, X[i]), y[i]))

        heap_size = len(h)
        label_list = []
        dist_list = []
        while heap_size > 0:
            dist, lab = heappop(h)
            print("at distant ", dist, " there is a point with label ", lab)
            heap_size -= 1
            if dist in dist_list:
                label_list.append(lab)
            else:
                if len(label_list) >= 3:
                    break
                label_list.append(lab)
                dist_list.append(dist)
        label_list = sorted(label_list)[0:3]
        print(majority_vote(label_list))


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
    # print("First split at threshold, ", threshold, " ; for attribute ", attribute)

    if threshold is not None:
        set_left = [x for x in X if x[attribute] <= threshold]
        y_left = [y[i] for i, x in enumerate(X) if x[attribute] <= threshold]
        set_right = [x for x in X if x[attribute] > threshold]
        y_right = [y[i] for i, x in enumerate(X) if x[attribute] > threshold]
        l_th, l_attr = split(set_left, y_left)
        r_th, r_attr = split(set_right, y_right)
        # print("Left node split at threshold, ", l_th, " ; for attribute ", l_attr)
        # print("Right node split at threshold, ", r_th, " ; for attribute ", r_attr)
        
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


# 1 0:1.5 2:7.1 3:7.8 4:1.1 5:8.1 9:5.1
# 1 0:7.0 2:1.1 3:1.2 4:0.1 5:5.1 9:2.1
# 1 0:1.0 2:1.1 3:8.4 4:6.1 5:8.1 9:7.1
# 2 0:5.0 2:1.1 3:7.2 4:5.1 5:8.1 9:8.1
# 1 0:1.0 2:1.1 3:8.4 4:6.1 5:8.1 9:7.1
# 8 0:2.0 2:2.1 3:7.2 4:5.1 5:2.1 9:8.1
# 8 0:2.0 2:1.1 3:8.4 4:5.1 5:8.1 9:7.1
# 8 0:5.0 2:1.1 3:7.2 4:5.1 5:5.1 9:7.1
# 2 0:1.0 2:4.1 3:11.2 4:2.4 5:8.1 9:4.1
# 2 0:1.0 2:8.1 3:20.2 4:3.1 5:8.1 9:6.1
# 3 0:1.0 2:1.1 3:4.2 4:2.5 5:4.1 9:8.6
# 3 0:1.0 2:1.1 3:2.2 4:8.1 5:1.1 9:7.7
# 0 0:7.0 2:2.2 3:1.2 4:2.2 5:8.1 9:2.5
# 0 0:4.5 2:5.0 3:10.2 4:7.1 5:2.5 9:6.8
# 0 0:0.5 2:1.0 3:2.2 4:3.1 5:1.5 9:7.3
# 0 0:1.5 2:1.0 3:7.2 4:8.1 5:4.5 9:3.6
# 0 0:7.0 2:7.2 3:1.2 4:2.2 5:8.1 9:2.5
# 0 0:4.5 2:5.0 3:5.2 4:7.1 5:8.5 9:7.8
# 0 0:0.5 2:2.0 3:2.2 4:3.1 5:7.5 9:4.3
# 0 0:1.5 2:1.0 3:2.2 4:8.1 5:4.5 9:3.6
# 1 0:1.0 2:1.0
# 1 0:1.0 2:2.0
# 1 0:2.0 2:1.0
# 3 0:2.0 2:2.0
# 1 0:3.0 2:1.0
# 3 0:3.0 2:2.0
# 3 0:3.0 2:3.0
# 3 0:4.5 2:3.0
# 0 0:1.0 2:2.2
# 0 0:4.5 2:1.0
filepath = 'dataset.txt'
inputs = []
with open(filepath) as fp:
   line = fp.readline()
   line = line.strip()
   inputs.append(line)
   while line:
       line = fp.readline()
       if len(line) > 0:
           line = line.strip()
           inputs.append(line)

train, label, test, data = read_inputs(inputs) # (sys.stdin)
decision_tree(train, label, test)
print()
# knn(train, label, test)
knn_kdtree(data, test)