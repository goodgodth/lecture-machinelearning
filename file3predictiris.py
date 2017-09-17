import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree

iris = load_iris()
test_idx = [0,5,10,20,30,40,60,80,100,120]

#Training Data
train_target = np.delete(iris.target, test_idx);
train_data = np.delete(iris.data, test_idx, axis = 0 )


#Testing data
test_target = iris.target[test_idx];
test_data = iris.data[test_idx];

dt = tree.DecisionTreeClassifier()
dt = dt.fit(train_data , train_target)
#dt = dt.fit(iris.data, iris.target);

print("Predicts: {}".format(dt.predict(test_data)))
print("Labels: {}".format(test_target))



