import numpy as
from sklearn.datasets import load_iris
from sklearn import neighbors

iris = load_iris();
test_idx = [0,5,10,20,30,40,60,80,100,120]

#Training Data
train_target = np.delete(iris.target, test_idx);
train_data = np.delete(iris.data, test_idx, axis = 0)

#Testing Data
test_target = iris.target[test_idx]
test_data = iris.data[test_idx]

knn = neighbors.KNeighborsClassifier(n_neighbors=3)
knn.fit(train_data,train_target)

print("Predicts: {}".format(knn.predict(test_data)))
print("Labels: {}".format(test_target))

