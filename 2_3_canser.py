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
