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

# %%
