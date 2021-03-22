import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_excel("C:/Users/user/Desktop/DATASETS/EastWestAirlines.xlsx", "data")
df.describe() #gives iqr and std values.
df.info()
df1 = df.drop(['ID#'],axis = 1)

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
plt.figure(figsize=(15,8));plt.title('h-clustering');plt.xlabel('index');plt.ylabel('distance')
sch.dendrogram(z);plt.show()


# Now applying AgglomerativeClustering choosing 3 as clusters from the above dendrogram
from sklearn.cluster import AgglomerativeClustering

h_complete = AgglomerativeClustering(n_clusters = 3, linkage = 'complete', affinity = "euclidean").fit(df_norm) 
h_complete.labels_

cluster_labels = pd.Series(h_complete.labels_)

df['cluster'] = cluster_labels # creating a new column and assigning it to new column 
df.describe()
df = df.iloc[:, [12,0,1,2,3,4,5,6,7,8,9,10,11]] #bring cluster column to 0th index
df.head()

# Aggregate mean of each cluster
df.iloc[:, 1:].groupby(df.cluster).mean()
# creating a excel file 
df.to_excel("EW_Airlines.csv", encoding = "utf-8")

import os
os.getcwd()
