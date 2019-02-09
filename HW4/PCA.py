import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from numpy import linalg as LA
from sklearn.metrics.pairwise import euclidean_distances


def unpickle(file):
    import _pickle as cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo, encoding='latin1')
    return dict


tags = unpickle('cifar-10-batches-py/batches.meta')
tags = tags['label_names']


# reading files
file_name = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5', 'test_batch']
mean = np.zeros((10,3072))
data = np.empty((0, 3072))
y = np.empty((0, 1))
class_index = {}
for fn in file_name:
    dict = unpickle('cifar-10-batches-py/'+fn)
    images = np.reshape(dict['data'], (10000, 3072))
    data = np.vstack((data, images))
    labels = dict['labels']
    l = np.array(labels).T.reshape(-1, 1)
    y = np.vstack((y, l))


# 1. mean image of each class
# also calculate index for every class for later use
for i in range(10):
    index = np.where(y == i)[0]
    class_index[i] = index
    m = np.mean(data[index], axis=0)
    mean[i] = m


mean_int = mean.astype(int)
plt.figure(figsize=(9, 9))
for i in range(10):
    plt.subplot(3, 4, i+1)
    img = np.reshape(mean_int[i], (3, -1)).T
    img = np.reshape(img, (32, 32, -1))
    plt.imshow(img)
    plt.title(tags[i])
plt.savefig('mean.png')
plt.clf()


# 2. plot of SSE with mean
pca = PCA(n_components=20)
lsse = []
for i in range(10):
    index = class_index[i]
    class_origin = data[index]
    class_model = pca.fit(class_origin)
    class_pca = class_model.transform(class_origin)
    class_pca = class_model.inverse_transform(class_pca)
    sse = np.mean(np.sum((class_origin-class_pca)**2))
    lsse.append(sse)
plt.figure(1, [10,5])
plt.bar(range(10), lsse, align='center')
plt.xticks(range(10), tags)
plt.savefig('sse.png')
plt.clf()


def pcoa(dis_map):
    A = np.eye(10) - np.ones((10, 10)) / 10
    W = -0.5 * np.dot(np.dot(A, dis_map), A.T)
    eig_val, eig_vec = LA.eig(W)
    index = np.argsort(eig_val)[-2:]
    eig_val = np.sqrt(eig_val[index])
    eig_val = np.diag([eig_val[1], eig_val[0]])
    print(eig_val)
    eig_vec_f = np.empty((10, 2))
    eig_vec_f[:, 0] = eig_vec[:, 1]
    eig_vec_f[:, 1] = eig_vec[:, 0]
    pcoa_y = np.dot(eig_vec_f, eig_val)
    return pcoa_y


# 3. Principle Coordinate Analysis with Euclidean
dis_map = np.square(euclidean_distances(mean, mean))
np.savetxt('partb_distances.csv', dis_map, delimiter=',')
pcoa_euc = pcoa(dis_map)
plt.scatter(pcoa_euc[:, 0], pcoa_euc[:, 1], marker='o')
for tag, x, y in zip(tags, pcoa_euc[:, 0], pcoa_euc[:, 1]):
    plt.annotate(tag, xy=(x, y))
plt.title("Part B PCoA")
plt.savefig('pcoa_b.png')
plt.clf()


# 4. Use similarity matrix
def ea2b(ind_i, ind_j):
    i_data = data[class_index[ind_i]]
    j_data = data[class_index[ind_j]]
    pca = PCA(n_components=20)
    model = pca.fit(j_data)
    diff_ij = model.inverse_transform(model.transform(i_data))
    sim_diff = np.mean(np.sum(np.square(diff_ij - i_data), axis=1))
    return sim_diff


dis_map = np.zeros((10, 10))
for i in range(10):
    for j in range(i, 10):
        diff = 0.5 * (ea2b(i, j) + ea2b(j, i))
        dis_map[i][j] = diff
        dis_map[j][i] = diff
np.savetxt('partc_distances.csv', dis_map, delimiter=',')

pcoa_sim = pcoa(dis_map)
plt.scatter(pcoa_sim[:, 0], pcoa_sim[:, 1], marker='o')
for tag, x, y in zip(tags, pcoa_sim[:, 0], pcoa_sim[:, 1]):
    plt.annotate(tag, xy=(x, y))
plt.title("Part C PCoA")
plt.savefig('pcoa_c.png')
plt.clf()
