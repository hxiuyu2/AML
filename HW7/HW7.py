import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_curve, auc
from sklearn.metrics.pairwise import cosine_similarity as cosdis
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression

data = pd.read_csv('yelp_2k.csv')
X = data['text']
y = data['stars'].astype(int)
X = [t.lower() for t in X]

# # Part 1.2: BOW w/o processing
# vec = CountVectorizer()
# vect_doc = vec.fit_transform(X).toarray()
# vocabulary = sorted(vec.vocabulary_.items(), key=lambda kv: kv[1])
# columns = [row[0] for row in vocabulary]
# df = pd.DataFrame(vect_doc, columns=columns)
# count = pd.DataFrame(df.sum(axis=0)).reset_index()
# count.columns = ['word', 'count']
# count = count.sort_values(by='count', ascending=False)
# plt.scatter(x=range(count.shape[0]), y=count['count'])
# plt.savefig('count_rank.png')


# Part 1.3: BOW with threshold
stop_word = ['the', 'and', 'to', 'was', 'it', 'of', 'for', 'in', 'my',
             'is', 'that', 'they', 'this', 'we', 'you', 'with', 'on',
             'have', 'had', 'me', 'at', 'so', 'were', 'are', 'be',
             'there', 'he', 'if', 'when', 'our', 'she', 'an', 'their', 'here',
             'will', 'about', 'them', 'your', 'us', 'been']
vec_stop = CountVectorizer(stop_words=stop_word, max_df=1000, min_df=5)
vect_doc = vec_stop.fit_transform(X).toarray()
vocabulary = sorted(vec_stop.vocabulary_.items(), key=lambda kv: kv[1])
columns = [row[0] for row in vocabulary]
df = pd.DataFrame(vect_doc, columns=columns)
count = pd.DataFrame(df.sum(axis=0)).reset_index()
count.columns = ['word', 'count']
count = count.sort_values(by='count', ascending=False)
plt.scatter(x=range(count.shape[0]), y=count['count'])
plt.savefig('count_rank_stop.png')
plt.clf()


# Part 2: find horrible customer service
target = vec_stop.transform(['horrible customer service']).toarray()
nbrs = NearestNeighbors(n_neighbors=5, algorithm='brute', metric='cosine')
nbrs.fit(vect_doc)
values, index = nbrs.kneighbors(target)
values = cosdis(vect_doc, target).flatten()
index = np.argsort(values)
values = np.sort(values)
for i,j in zip(index, values):
    review = X[i][:200] if len(X[i]) > 200 else X[i]
    print("------------------ for score ", j, " at index ", i, " ---------------------")
    print(review)


# # Part 3: regression
clf = LogisticRegression(solver='lbfgs')
rows = vect_doc.shape[0]
index = np.arange(0, rows)
np.random.shuffle(index)
test_index = index[:int(0.1 * rows)]
train_index = index[int(0.1 * rows):]
train_X = pd.DataFrame(vect_doc).iloc[train_index]
train_y = pd.DataFrame(y).iloc[train_index]
test_X = pd.DataFrame(vect_doc).iloc[test_index]
test_y = pd.DataFrame(y).iloc[test_index]


clf.fit(train_X, train_y)
predictions = clf.predict(train_X)
# predictions = clf.predict(test_X)


# # Part 3: histogram
train_prob = clf.predict_proba(train_X)
positive = [prob[0] for prob in train_prob[predictions == 1]]
negative = [prob[0] for prob in train_prob[predictions == 5]]
plt.hist(positive, bins=100)
plt.hist(negative, bins=100)
plt.savefig('hist_prob.png')
plt.clf()


# Part 3: new threshold
train_prob = clf.predict_proba(train_X)
predictions = [1 if prob[0] > 0.6 else 5 for prob in train_prob]
correct = 0
values = train_y['stars'].tolist()
for pred, val in zip(predictions, values):
    if pred == val:
        correct += 1
print('train accuracy is ', correct/0.9/rows)

train_prob = clf.predict_proba(test_X)
predictions = [1 if prob[0] > 0.6 else 5 for prob in train_prob]
correct = 0
values = test_y['stars'].tolist()
for pred, val in zip(predictions, values):
    if pred == val:
        correct += 1
print('test accuracy is ', correct/0.1/rows)


# Part 3: ROC plot
test_prob = clf.predict_proba(test_X)
y_true = [0 if y == 1 else 1 for y in test_y.values]
y_score = [proba[1] for proba in test_prob]
fpr, tpr, threshold = roc_curve(y_true, y_score)
auc = auc(fpr, tpr)
plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.2f)' % auc)
plt.legend(loc="lower right")
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.savefig('roc.png')
plt.clf()


# Part 3: find optimal threshold
min_distance = 1000
result = (0,0,0)
for f,t,th in zip(fpr, tpr, threshold):
    distance = np.sqrt(np.square(f) + np.square(1-t))
    if distance < min_distance:
        result = (f, t, th)
        min_distance = distance

print('at the threshold ', result[2], ', the best (FPR, TPR) is (',result[0],', ',result[1],')')
