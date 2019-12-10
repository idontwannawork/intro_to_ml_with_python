# %% [markdown]
# ## Cancer data

# %%
from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from IPython.display import display
import mglearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

# %%
cancer = load_breast_cancer()
print('cancer.keys(): \n{}'.format(cancer.keys()))

# %%
print('Shape of cancer data: \n{}'.format(cancer.data.shape))

# %%
print('Sample counts per class: \n{}'.format(
    {n: v for n, v in zip(cancer.target_names, np.bincount(cancer.target))}))
# %%
print('Feature names: \n{}'.format(cancer.feature_names))

# %%
print('Description: \n{}'.format(cancer.DESCR))

# %%
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, stratify=cancer.target, random_state=66)

training_accuracy = []
test_accuracy = []
neighbors_settings = range(1, 11)

for n_neighbors in neighbors_settings:
    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(X_train, y_train)
    training_accuracy.append(clf.score(X_train, y_train))
    test_accuracy.append(clf.score(X_test, y_test))

plt.plot(neighbors_settings, training_accuracy, label='training accuracy')
plt.plot(neighbors_settings, test_accuracy, label='test accuracy')
plt.xlabel('n_neighbors')
plt.ylabel('accuracy')
plt.legend()

# %%
