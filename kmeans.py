import matplotlib.pyplot as plt
import numpy as np
from matplotlib import style
style.use("ggplot")
from sklearn.cluster import KMeans

x = [60,20,35,21,15,55,32]
y = [1.1,8.2,4.2,1.5,7.6,2.0,3.9]
colors = ['g.','r.','c.','y.']
plt.subplot(2, 1, 1)
plt.scatter(x,y)
plt.ylabel("y values")
X=np.array([[60, 1.1], [20, 8.2], [35, 4.2],[21, 1.5], [15, 7.6], [55, 2.0],[32,3.9]]) # book example
k=4
kmeans = KMeans(n_clusters=k)
kmeans.fit(X)
centroids=kmeans.cluster_centers_
labels=kmeans.labels_

plt.subplot(2, 1, 2)
for i in range(len(X)):
    plt.plot(X[i][0],X[i][1],colors[labels[i]],markersize=10)
for i in range(k):
    print("Cluster {0} = ".format(i+1),end=' ')
    for j in range(len(X)):
        if labels[j]==i:
            print(X[j],end=' , ')
    print()
plt.scatter(centroids[:,0],centroids[:,1],marker='x',s=150,linewidths=5,zorder=10)
plt.xlabel("x values")
plt.ylabel("y values")
plt.savefig('result.png', bbox_inches='tight')
plt.show()

# Sara Mazaheri
#Kmeans Algorithms
#Data Mining
