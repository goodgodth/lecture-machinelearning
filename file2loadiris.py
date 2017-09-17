from sklearn.datasets import load_iris

iris = load_iris()

print(iris.feature_names);
print(iris.target_names);
print(iris.data[0]);
print(iris.target[0]);
print("++++++++++++++");


for i in range(len(iris.target)):
    print("{}\tlabel:{}\t\tfeature:{}".format(i+1, iris.target[i], iris.data[i]))