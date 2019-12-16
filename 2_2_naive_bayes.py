# %% [markdown]
# ## Naive bayes

from sklearn.model_selection import train_test_split
from IPython.display import display
import mglearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

# %%
X = np.array([[0, 1, 0, 1],
              [1, 0, 1, 1],
              [0, 0, 0, 1],
              [1, 0, 1, 0]])
y = np.array([0, 1, 0, 1])

# %%
counts = {}
for label in np.unique(y):
    counts[label] = X[y == label].sum(axis=0)

print('Feature counts: {}'.format(counts))

# %%
