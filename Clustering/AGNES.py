import sys
from heapq import heappush, heappop

# read input
def read_inputs(ins):
    line_number = 0
    points = []
    N = 0
    k = 0
    for line in ins:
        if line_number == 0:
            N, k = line.split()
            N = int(N)
            k = int(k)
        elif line_number <= int(N):
            parts = line.split()
            results = list(map(float, parts))
            points.append(results)
        else:
            break
        line_number += 1
    return points, N, k


def euclidean(a, b):
    diff = [x - y for x, y in zip(a, b)]
    diff = [num ** 2 for num in diff]
    return sum(diff)


def get_distance_matrix(data):
    h = []
    for index, row in enumerate(data):
        for ind in range(index+1, len(data)):
            distance = euclidean(data[ind], row)
            heappush(h, (distance, [index, ind]))
    return h


def agnes(points, N, k):
    dist_heap = get_distance_matrix(points)
    clusters = []
    same_dist = []
    first = 0
    i = 0
    same_first = []
    # visited = set()
    while i < (N - k):
        if len(same_first) > 0:
            ind0 = first
            ind1 = heappop(same_first)
        elif len(same_dist) > 0:
            ind0, ind1 = heappop(same_dist)
            while len(same_dist) > 0:
                next_ind, _ = same_dist[0]
                _, second = heappop(same_dist)
                heappush(same_first, second)
                if next_ind != ind0:
                    first = ind0
                    break
        else:
            cur_dist, pair = heappop(dist_heap)
            while len(dist_heap) > 0:
                dist, _ = dist_heap[0]
                _, p = heappop(dist_heap)
                heappush(same_dist, (p[0], p[1]))
                if cur_dist != dist:
                    break
            ind0 = pair[0]
            ind1 = pair[1]

        assigned = [-1, -1]
        for j in range(len(clusters)):
            if assigned[0] != -1 and assigned[1] != -1:
                break
            if ind0 in clusters[j]:
                assigned[0] = j
            if ind1 in clusters[j]:
                assigned[1] = j

        if assigned[0] == -1 and assigned[1] == -1:
            temp_set = set()
            temp_set.add(ind0)
            temp_set.add(ind1)
            clusters.append(temp_set)
        elif assigned[0] == -1:
            clusters[assigned[1]].add(ind0)
        elif assigned[1] == -1:
            clusters[assigned[0]].add(ind1)
        else:
            if assigned[0] == assigned[1]:
                i -= 1
            else:
                i1 = max(assigned)
                i2 = min(assigned)
                set0 = clusters[i1]
                set1 = clusters[i2]
                set1.update(set0)
                del clusters[i1]
        i += 1

    res = [-1] * N
    for i in range(len(clusters)):
        label = min(clusters[i])
        for ind in clusters[i]:
            res[ind] = label

    return res


# inputs = ['10 2',
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
points, N, k = read_inputs(inputs) # (sys.stdin)
res = agnes(points, N, k)
for index, r in enumerate(res):
    if r == -1:
        print(index)
    else:
        print(r)