import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv("C:/Users/user/Desktop/DATASETS/crime_data.csv")
df.describe() #gives iqr and std values.
df.info()
df1 = df.drop(['Unnamed: 0'],axis = 1)

#creating normalization function             
def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return (x)

# Normalized data frame (considering the numerical part of data)
df_norm = norm_func(df1)
df_norm.describe()


# for creating dendrogram 
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch

z = linkage(df_norm, method='single',metric= 'euclidean')

#dendrogram
plt.figure(figsize=(14,6));plt.title('h-clustering');plt.xlabel('index');plt.ylabel('distance')
sch.dendrogram(z);plt.show()


# Now applying AgglomerativeClustering choosing 4 as clusters from the above dendrogram
from sklearn.cluster import AgglomerativeClustering

h_complete = AgglomerativeClustering(n_clusters = 4, linkage = 'complete', affinity = "euclidean").fit(df_norm);plt.show() 
h_complete.labels_

cluster_labels = pd.Series(h_complete.labels_)

df['cluster'] = cluster_labels # creating a new column and assigning it to new column 
df.describe()
df = df.iloc[:, [5,0,1,2,3,4]] #bring cluster column to 0th index
df.head()

# Aggregate mean of each cluster
df.iloc[:, 2:].groupby(df.cluster).mean()
# creating a csv file 
df.to_csv("crimedata.csv", encoding = "utf-8")

import os
os.getcwd()
