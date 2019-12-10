# %% [markdown]
# ## Boston housing

# %%
from sklearn.datasets import load_boston
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
boston = load_boston()
print('Data shape: \n{}'.format(boston.data.shape))

# %%
X, y = mglearn.datasets.load_extended_boston()
print('X.shape: \n{}'.format(X.shape))

# %% [markdown]
# ## k-NN algorithm

# %%
mglearn.plots.plot_knn_classification(n_neighbors=1)

# %%
mglearn.plots.plot_knn_classification(n_neighbors=3)

# %%
X, y = mglearn.datasets.make_forge()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# %%
clf = KNeighborsClassifier(n_neighbors=3)

# %%
clf.fit(X_train, y_train)

# %%
print('Test set predictions: {}'.format(clf.predict(X_train)))

# %%
print('Test set accuracy: {:.2f}'.format(clf.score(X_test, y_test)))

# %% [markdown]
# ## Display decision boundary


# %%
fig, axes = plt.subplots(1, 3, figsize=(10, 3))
for n_neighbors, ax in zip([1, 3, 9], axes):
    clf = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X, y)
    mglearn.plots.plot_2d_separator(
        clf, X, fill=True, eps=0.5, ax=ax, alpha=.4)
    mglearn.discrete_scatter(X[:, 0], X[:, 1], y, ax=ax)
    ax.set_title('{} neighbor(s)'.format(n_neighbors))
    ax.set_xlabel('feature 0')
    ax.set_ylabel('feature 1')

axes[0].legend(loc=3)

# %%
