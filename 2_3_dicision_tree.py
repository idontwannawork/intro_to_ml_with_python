# %% [markdown]
# ## Dicision tree

# %%
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
import os
import graphviz
from sklearn.tree import export_graphviz
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from IPython.display import display
import mglearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline


# %%
mglearn.plots.plot_animal_tree()

# %%

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, stratify=cancer.target, random_state=42)
tree = DecisionTreeClassifier(random_state=0)
tree.fit(X_train, y_train)
print('Accuracy on training set: {:.3f}'.format(tree.score(X_train, y_train)))
print('Accuracy on test set: {:.3f}'.format(tree.score(X_test, y_test)))

# %%
tree = DecisionTreeClassifier(max_depth=4, random_state=0)
tree.fit(X_train, y_train)
print('Accuracy on training set: {:.3f}'.format(tree.score(X_train, y_train)))
print('Accuracy on test set: {:.3f}'.format(tree.score(X_test, y_test)))

# %%
export_graphviz(tree, out_file='tree.dot', class_names=[
                'malignant', 'benign'], feature_names=cancer.feature_names, impurity=False, filled=True)

# %%

with open('tree.dot') as f:
    dot_graph = f.read()

graphviz.Source(dot_graph)

# %%
print('Feature importance: \n{}'.format(tree.feature_importances_))


# %%
def plot_feature(model):
    n_features = cancer.data.shape[1]
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), cancer.feature_names)
    plt.xlabel('Feature importance')
    plt.ylabel('Feature')


plot_feature(tree)

# %%
tree = mglearn.plots.plot_tree_not_monotone()
display(tree)


# %%

ram_prices = pd.read_csv(os.path.join(
    mglearn.datasets.DATA_PATH, 'ram_price.csv'))

plt.semilogy(ram_prices.date, ram_prices.price)
plt.xlabel('Year')
plt.ylabel('Price in $/Mbyte')


# %%

train = ram_prices[ram_prices.date < 2000]
test = ram_prices[ram_prices >= 2000]

X_train = train.date[:, np.newaxis]
y_train = np.log(train.price)

tree = DecisionTreeRegressor().fit(X_train, y_train)
linear = LinearRegression().fit(X_train, y_train)

X_all = ram_prices.date[:, np.newaxis]

pred_tree = tree.predict(X_all)
pred_lr = linear.predict(X_all)

price_tree = np.exp(pred_tree)
price_lr = np.exp(pred_lr)


# %%
plt.semilogy(train.date, train.price, label='Training data')
plt.semilogy(test.date, test.price, label='Test data')
plt.semilogy(ram_prices.date, price_tree, label='Tree prediction')
plt.semilogy(ram_prices.date, price_lr, label='Linear prediction')
plt.legend()

# %%
