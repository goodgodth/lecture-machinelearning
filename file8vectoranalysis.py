from sklearn import decomposition #การจำแนก
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

iris = load_iris()

pca = decomposition.PCA(n_components=2)
pca.fit(iris.data)

x = pca.transform(iris.data)


# plt.scatter(x[:,0], x[:,1])
# plt.show()

# plt.scatter(x[:,0], x[:,1], c=iris.target)
# plt.show()

km = KMeans(3)
km.fit(x)

print(km.cluster_centers_)

plt.scatter(x[:,0] , x[:,1], c=iris.target)
plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:,1], marker='x', s=300, c="red")

plt.show()
