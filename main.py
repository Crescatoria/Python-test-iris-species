from sklearn import datasets

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

import matplotlib.pyplot as plt
import seaborn as sns

iris = datasets.load_iris()

data = iris.data
target = iris.target

feature_names = iris.feature_names
target_names = iris.target_names

print("Data Shape: ", data.shape)
print("Feature names: ", feature_names)
print("Target names: ", target_names)

X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)

clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)
print("Accuracy:", accuracy)

sns.pairplot(sns.load_dataset("iris"), hue="species")
plt.show()