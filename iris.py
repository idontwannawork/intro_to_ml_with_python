# %% [markdown]
# ## Class iris

# %%

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from IPython.display import display
import mglearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline


# %% [markdown]
# ## Load iris data from scikit-learn

# %%


iris_dataset = load_iris()

# %%

print('Keys of iris_dataset: \n{}'.format(iris_dataset.keys()))


# %% [markdown]
# ## Print description of iris dataset

# %%

print(iris_dataset['DESCR'][:300] + '\n...')

# %% [markdown]
# ## Print target names of iris

# %%

print('Target names: {}'.format(iris_dataset['target_names']))

# %% [markdown]
# ## Print feature name of iris

# %%
print('Feature names: \n{}'.format(iris_dataset['feature_names']))


# %% [markdown]
# ## Print type of iris data

# %%

print('Type of data: {}'.format(type(iris_dataset['data'])))

# %% [markdown]
# ## Print shape of iris data

# %%
print('Shape of data: {}'.format(iris_dataset['data'].shape))

# %% [markdown]
# ## Print first columns of data

# %%
print('Shape of data: \n{}'.format(iris_dataset['data'][:5]))

# %% [markdown]
# Print type of target data

# %%
print('Type of targer: \n{}'.format(type(iris_dataset['target'])))

# %% [markdown]
# ## Print shape of target

# %%
print('Shape of target: \n{}'.format(iris_dataset['target']))

# %% [markdown]
# ## Generate and generalize model

# %%
X_train, X_test, y_train, y_test = train_test_split(
    iris_dataset['data'], iris_dataset['target'], random_state=0)

print('X_train shape: {}'.format(X_train.shape))
print('y_train shape: {}'.format(y_train.shape))

print('X_test shape: {}'.format(X_test.shape))
print('y_test shape: {}'.format(y_test.shape))

# %% [markdown]
# ## Display data scatter matrix

# %%

iris_df = pd.DataFrame(X_train, columns=iris_dataset.feature_names)
grr = pd.plotting.scatter_matrix(iris_df, c=y_train, figsize=(
    15, 15), marker='o', hist_kwds={'bins': 20}, s=60, alpha=.8, cmap=mglearn.cm3)

# %% [markdown]
# ## Import k-Nearest Neighbors

# %%
knn = KNeighborsClassifier(n_neighbors=1)

# %%
knn.fit(X_train, y_train)

# %% [markdown]
# ## Predict iris

# %%
X_new = np.array([[5, 2.9, 1, 0.2]])
print('X_new.shape: {}'.format(X_new.shape))

prediction = knn.predict(X_new)
print('Prediction: {}'.format(prediction))
print('Predicted target name: {}'.format(
    iris_dataset['target_names'][prediction]))

# %% [markdown]
# ## Evaluate model


# %%
y_pred = knn.predict(X_test)
print('Test set predictions: \n{}'.format(y_pred))
print('Test set score: {:.2f}'.format(np.mean(y_pred == y_test)))
print('Test set score: {:.2f}'.format(knn.score(X_test, y_test)))

# %% [markdown]
# ## Train and evaluate model to the minimum

# %%
# X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)

# knn = KNeighborsClassifier(n_neighbors=1)
# knn.fit(X_train, y_train)

# print('Test set score: {:2f}'.format(knn.score(X_test, y_test)))
