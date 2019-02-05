from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

# train PCA on clean data
iris = pd.read_csv('hw3-data/iris.csv')
clean_models = [None]*5
for i in range(1, 5):
    clean_models[i] = PCA(n_components=i) # whiten = False (default)
    clean_models[i].fit(iris)


# sum of mean square error
def mse(noisy):
    column = len(iris.columns)
    mse = mean_squared_error(noisy, iris)
    mse = mse * column
    return mse


# read data
file_name = ['I', 'II', 'III', 'IV', 'V']
result = pd.DataFrame(columns=['0N', '1N', '2N', '3N', '4N', '0c', '1c', '2c', '3c', '4c'],
                      index=['Dataset I', 'Dataset II', 'Dataset III', 'Dataset IV', 'Dataset V'])
for fn in file_name:
    data = pd.read_csv('hw3-data/data' + fn + '.csv')
    mean = data.mean(axis=0).values
    zero_noisy = np.repeat(mean, data.shape[0]).reshape(4,-1)
    c_mse = mse(zero_noisy.T)
    print('Dataset ' + str(fn) + ' 0c', c_mse)
    result.loc['Dataset ' + str(fn), '0c'] = c_mse
    for i in range(1, 5):
        noisy_model = PCA(n_components=i)  # whiten = False (default)
        noisy_model.fit(data)
        pca_data = noisy_model.transform(data)
        pca_data = noisy_model.inverse_transform(pca_data)
        c_mse = mse(pca_data)
        print('Dataset '+str(fn), str(i)+'c', c_mse)
        result.loc['Dataset '+str(fn), str(i)+'c'] = c_mse

        if (fn == 'I') & (i == 2):
            data1 = pd.DataFrame(pca_data)
            data1.to_csv('dataI.csv')

        pca_data = clean_models[i].transform(data)
        pca_data = clean_models[i].inverse_transform(pca_data)
        n_mse = mse(pca_data)
        print('Dataset ' + str(fn), str(i) + 'N', c_mse)
        result.loc['Dataset ' + str(fn), str(i) + 'N'] = n_mse


mean = iris.mean(axis=0).values
zero_clean = np.repeat(mean, iris.shape[0]).reshape(4,-1)
n_mse = mse(zero_clean.T)
print('Dataset ' + str(fn) + ' 0N', n_mse)
result['0N'] = n_mse

result.to_csv('hxiuyu2-numbers.csv', index=False)