import pandas as pd 
import numpy as np
wine = pd.read_csv("C:/Training/Analytics/PCA/wine/wine.csv")
wine.describe()
wine.head()

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale 

# Considering only numerical data 
wine.data = wine.ix[:,1:]
wine.data.head(4)

# Normalizing the numerical data 
wine_normal = scale(wine.data)

pca = PCA(n_components = 13)
pca_values = pca.fit_transform(wine_normal)


# The amount of variance that each PCA explains is 
var = pca.explained_variance_ratio_
var
pca.components_[0]

# Cumulative variance 

var1 = np.cumsum(np.round(var,decimals = 4)*100)
var1

# =============================================================================
# # Variance plot for PCA components obtained 
# plt.plot(var1,color="red")
# 
# # plot between PCA1 and PCA2 
# x = pca_values[:,0]
# y = pca_values[:,1]
# z = pca_values[:2:3]
# plt.scatter(x,y,color=["red","blue"])
# plt.scatter(x,y,color=['red','blue'])
# 
# from mpl_toolkits.mplot3d import Axes3D
# Axes3D.scatter(np.array(x),np.array(y),np.array(z),c=["green","blue","red"])
# 
# 
# =============================================================================
################### Clustering  ##########################
new_df = pd.DataFrame(pca_values[:,0:3])

from	sklearn.cluster	import	KMeans
from scipy.spatial.distance import cdist
###### screw plot or elbow curve ############
k = list(range(2,15))
k
TWSS = [] # variable for storing total within sum of squares for each kmeans 
for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(new_df)
    WSS = [] # variable for storing within sum of squares for each cluster 
    for j in range(i):
        WSS.append(sum(cdist(new_df.iloc[kmeans.labels_==j,:],kmeans.cluster_centers_[j].reshape(1,new_df.shape[1]),"euclidean")))
    TWSS.append(sum(WSS))


# Scree plot 
plt.plot(k,TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS");plt.xticks(k)

kmeans = KMeans(n_clusters = 3)
kmeans.fit(new_df)
klables = kmeans.labels_

wine['clust_kmeans']=klables # creating a  new column and assigning it to new column 
wine = wine.iloc[:,[14,0,1,2,3,4,5,6,7,8,9,10,11,12,13]]
wine.head()

wine["Type"].replace({1:0, 2:1, 3:2}, inplace=True)


from sklearn.metrics import confusion_matrix
confusion_matrix1 = confusion_matrix(wine['clust_kmeans'],wine['Type'])
print (confusion_matrix1)
pd.crosstab(wine['Type'],wine['clust_kmeans'])
Y = wine.iloc[:,1]
accuracy = sum(Y==wine['clust_kmeans'])/wine.shape[0]

# =============================================================================
# ###################### Hierarchial clustering ################## 
# =============================================================================

from scipy.cluster.hierarchy import linkage 

import scipy.cluster.hierarchy as sch # for creating dendrogram 

type(new_df)

#p = np.array(new_df) # converting into numpy array format 
help(linkage)
z = linkage(new_df, method="complete",metric="euclidean")

plt.figure(figsize=(15, 5));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(
    z,
    leaf_rotation=0.,  # rotates the x axis labels
    leaf_font_size=8.,  # font size for the x axis labels
)
plt.show()

help(linkage)

# Now applying AgglomerativeClustering choosing 3 as clusters from the dendrogram
from	sklearn.cluster	import	AgglomerativeClustering 
h_complete	=	AgglomerativeClustering(n_clusters=3,	linkage='complete',affinity = "euclidean").fit(new_df) 


cluster_labels=pd.Series(h_complete.labels_)

wine['hie_clust']=cluster_labels # creating a  new column and assigning it to new column 
wine = wine.iloc[:,[15,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]]
wine.head()

# getting aggregate mean of each cluster
wine.iloc[:,2:].groupby(wine.hie_clust).median()

#confusion matrix and accuracy ##########
confusion_matrix2 = confusion_matrix(wine['hie_clust'],wine['Type'])
print (confusion_matrix2)
pd.crosstab(wine['Type'],wine['hie_clust'])
Y = wine.iloc[:,1]
accuracy = sum(Y==wine['hie_clust'])/wine.shape[0]