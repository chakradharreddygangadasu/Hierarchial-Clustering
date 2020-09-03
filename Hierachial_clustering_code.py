## Hierarchial clustering

"""The mall data set consists of independent variables 'CustomerID', 'Genre', 'Age', 'Annual Income (k$)',
'Spending Score (1-100)' so based on the available features we need to make clusters with hierarchial clustering 
that would help the organization for the development. Since there is no dependent variable its called 
unsupervised learning. """

# importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#importing dataset
dataset = pd.read_csv('C:/Users/gchak/Desktop/Machine Learning A-Z Template Folder/Part 4 - Clustering/Section 25 - Hierarchical Clustering/P14-Hierarchical-Clustering/Hierarchical_Clustering/Mall_Customers.csv')
x = dataset.iloc[:,[3,4]].values

## Here to make clusters we need to know the optimum number of clusters, and that can be found by dindrogram
## importing libraries for dindrogram
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(x, method = 'ward'))
plt.title('dindrogram')
plt.xlabel('customers')
plt.ylabel('euclidean distance')
plt.show()

## here we need to decide where to cut the dendrogram vertical line(it shows the distance between the clusters)
## the selection is made such that the maximum length of the vertival bars(makes sure both the legs of the bar)
## fro the graph blue bar have max length so, takin the threashold between 100 to 240 is a good one.
## Therefore it gives the number of clusters = 5 

## fitting the model with 5 clusters
##we are taking the commonly used 'agglomerative hierarchial clustering'

##creating and fiiting the model

from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')
y_hc = hc.fit_predict(x)

## visualizing the plots
plt.scatter(x[y_hc == 0,0], x[y_hc == 0,1], s = 100, color = 'red', label = 'cluster1')
plt.scatter(x[y_hc == 1, 0], x[y_hc ==1, 1], s =100, color = 'blue', label = 'cluster2')
plt.scatter(x[y_hc ==2, 0], x[y_hc == 2, 1], s = 100, color = 'green', label = 'cluster3')
plt.scatter(x[y_hc == 3, 0], x[y_hc ==3,1], s = 100, color = 'cyan', label = 'cluster4')
plt.scatter(x[y_hc == 4, 0], x[y_hc == 4, 1], s = 100, color = 'magenta', label = 'cluster5')
plt.title('Agglomearative jierarchial clustering')
plt.xlabel('customers')
plt.ylabel('euclidean distance')
plt.legend()
plt.show()















