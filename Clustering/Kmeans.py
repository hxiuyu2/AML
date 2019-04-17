import sys


# read input
def read_inputs(ins):
    line_number = 0
    points = []
    centriods = []
    N = 0
    for line in ins:
        if line_number == 0:
            N, k = line.split()
        elif line_number <= int(N):
            parts = line.split()
            results = list(map(float, parts))
            points.append(results)
        else:
            centriods.append(list(map(float, line.split())))
        line_number += 1
    return points, centriods


def euclidean(a, b):
    diff = [x - y for x, y in zip(a, b)]
    diff = [num ** 2 for num in diff]
    return sum(diff)


def get_new_center(cluster):
    row_count = len(cluster)
    axis_count = len(cluster[0])
    new_center = [0] * axis_count
    for row in cluster:
        for axis in range(axis_count):
            new_center[axis] += row[axis] / row_count
    return new_center


def kmeans(points, centroids):
    converged = len(centroids)
    while converged > 0:
        center_dict = {}
        labels = []
        print(centroids)

        for p in points:
            min_dist = 1000000000
            min_ind = 0
            for ind, c in enumerate(centroids):
                distance = euclidean(p, c)
                if distance < min_dist:
                    min_dist = distance
                    min_ind = ind
            if min_ind in center_dict:
                temp = center_dict[min_ind]
                temp.append(p)
            else:
                center_dict[min_ind] = [p]
            labels.append(min_ind)

        converged = len(centroids)
        temp_center = [None] * converged
        print(center_dict)
        for label, cluster in center_dict.items():
            new_center = get_new_center(cluster)
            temp_center[label] = new_center
            if new_center == centroids[label]:
                converged -= 1
        centroids = temp_center
    return labels


# ins = ['10 2',
# '8.98320053625 -2.08946304844',
# '2.61615632899 9.46426282022',
# '1.60822068547 8.29785986996',
# '8.64957587261 -0.882595891607',
# '1.01364234605 10.0300852081',
# '1.49172651098 8.68816850944',
# '7.95531802235 -1.96381815529',
# '0.527763520075 9.22731148332',
# '6.91660822453 -3.2344537134',
# '6.48286208351 -0.605353440895',
# '3.35228193353 6.27493570626',
# '6.76656276363 6.54028732984']
filepath = 'data4.txt'
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
points, centroids = read_inputs(inputs) # (sys.stdin)
res = kmeans(points, centroids)
for r in res:
    print(r)