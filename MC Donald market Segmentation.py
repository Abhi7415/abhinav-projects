#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


dfmd = pd.read_csv(r"C:\Users\Abhinav\Desktop\mcdonalds.csv")


# In[3]:


dfmd


# In[5]:


dfmd.sample(5)


# In[6]:


dfmd.isna().sum()


# In[7]:


dfmd.shape


# In[8]:


dfmd.info()


# In[9]:


dfmd["Gender"].value_counts()


# In[10]:


dfmd["VisitFrequency"].value_counts()


# In[11]:


dfmd["Like"].value_counts()


# In[12]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[18]:


plt.pie(dfmd['Gender'].value_counts(),labels=['Female','Male'],autopct='%0.1f%%',pctdistance=0.85)
plt.show()


# In[19]:


import seaborn as sns


# In[20]:


plt.figure(figsize=(25,8))
sns.countplot(x="Age",data=dfmd,palette='hsv')
plt.grid()


# In[23]:


dfmd['Like']= dfmd['Like'].replace({'I hate it!-5': '-5','I love it!+5':'+5'})
 
sns.barplot(x="Like", y="Age",data=dfmd,)
plt.title('Likelyness of McDonald w.r.t Age')
plt.show()


# In[24]:


from sklearn.preprocessing import LabelEncoder
def encoding(x):
    dfmd[x] = LabelEncoder().fit_transform(dfmd[x])
    return dfmd

category = ['yummy', 'convenient', 'spicy', 'fattening', 'greasy', 'fast', 'cheap',
       'tasty', 'expensive', 'healthy', 'disgusting']

for i in category:
    encoding(i)
dfmd


# In[25]:


dfmd


# In[26]:


#Considering only first 11 attributes
df_eleven = dfmd.loc[:,category]
df_eleven


# In[27]:


#Considering only the 11 cols and converting it into array
x = dfmd.loc[:,category].values
x


# In[28]:


#Principal component analysis

from sklearn.decomposition import PCA
from sklearn import preprocessing

pca_data = preprocessing.scale(x)

pca = PCA(n_components=11)
pc = pca.fit_transform(x)
names = ['pc1','pc2','pc3','pc4','pc5','pc6','pc7','pc8','pc9','pc10','pc11']
dfpc = pd.DataFrame(data = pc, columns = names)
dfpc


# In[29]:


#Proportion of Variance (from PC1 to PC11)
pca.explained_variance_ratio_


# In[30]:


np.cumsum(pca.explained_variance_ratio_)


# In[31]:


# correlation coefficient between original variables and the component

loadings = pca.components_
num_pc = pca.n_features_
pc_list = ["PC"+str(i) for i in list(range(1, num_pc+1))]
loadings_df = pd.DataFrame.from_dict(dict(zip(pc_list, loadings)))
loadings_df['variable'] = df_eleven.columns.values
loadings_df = loadings_df.set_index('variable')
loadings_df


# In[32]:


#Correlation matrix plot for loadings 
plt.rcParams['figure.figsize'] = (20,15)
ax = sns.heatmap(loadings_df, annot=True, cmap='Spectral')
plt.show()


# In[36]:


pip install bioinfokit


# In[37]:


#Scree plot (Elbow test)- PCA
from bioinfokit.visuz import cluster
cluster.screeplot(obj=[pc_list, pca.explained_variance_ratio_],show=True,dim=(10,5))


# In[39]:


# get PC scores
pca_scores = PCA().fit_transform(x)

# get 2D biplot
cluster.biplot(cscore=pca_scores, loadings=loadings, labels=dfmd.columns.values, var1=round(pca.explained_variance_ratio_[0]*100, 2),
    var2=round(pca.explained_variance_ratio_[1]*100, 2),show=True,dim=(10,5))


# In[40]:


import warnings
warnings.filterwarnings('ignore')


# In[46]:


get_ipython().system('pip install yellowbrick')


# In[61]:


#K-means clustering 

kmeans = KMeans(n_clusters=4, init='k-means++', random_state=0).fit(df_eleven)
dfmd['cluster_num'] = kmeans.labels_ #adding to df
print (kmeans.labels_) #Label assigned for each data point
print (kmeans.inertia_) #gives within-cluster sum of squares. 
print(kmeans.n_iter_) #number of iterations that k-means algorithm runs to get a minimum within-cluster sum of squares
print(kmeans.cluster_centers_) #Location of the centroids on each cluster. 


# In[62]:


#To see each cluster size
from collections import Counter
Counter(kmeans.labels_)


# In[63]:


#Visulazing clusters
sns.scatterplot(data=dfpc, x="pc1", y="pc2", hue=kmeans.labels_)
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], 
            marker="X", c="r", s=80, label="centroids")
plt.legend()
plt.show()


# In[65]:


#DESCRIBING SEGMENTS

from statsmodels.graphics.mosaicplot import mosaic
from itertools import product

crosstab =pd.crosstab(dfmd['cluster_num'],dfmd['Like'])
#Reordering cols
crosstab = crosstab[['-5','-4','-3','-2','-1','0','+1','+2','+3','+4','+5']]
crosstab 


# In[66]:


#Mosaic plot gender vs segment
crosstab_gender =pd.crosstab(dfmd['cluster_num'],dfmd['Gender'])
crosstab_gender


# In[67]:


plt.rcParams['figure.figsize'] = (7,5)
mosaic(crosstab_gender.stack())
plt.show()


# In[68]:


#box plot for age

sns.boxplot(x="cluster_num", y="Age", data=dfmd)


# In[69]:


#Calculating the mean
#Visit frequency
dfmd['VisitFrequency'] = LabelEncoder().fit_transform(dfmd['VisitFrequency'])
visit = dfmd.groupby('cluster_num')['VisitFrequency'].mean()
visit = visit.to_frame().reset_index()
visit


# In[70]:


dfmd['Like'] = LabelEncoder().fit_transform(dfmd['Like'])
Like = dfmd.groupby('cluster_num')['Like'].mean()
Like = Like.to_frame().reset_index()
Like


# In[71]:


#Gender
dfmd['Gender'] = LabelEncoder().fit_transform(dfmd['Gender'])
Gender = dfmd.groupby('cluster_num')['Gender'].mean()
Gender = Gender.to_frame().reset_index()
Gender


# In[72]:


segment = Gender.merge(Like, on='cluster_num', how='left').merge(visit, on='cluster_num', how='left')
segment


# In[73]:


plt.figure(figsize = (9,4))
sns.scatterplot(x = "VisitFrequency", y = "Like",data=segment,s=400, color="r")
plt.title("Simple segment evaluation plot for the fast food data set",
          fontsize = 15) 
plt.xlabel("Visit", fontsize = 12) 
plt.ylabel("Like", fontsize = 12) 
plt.show()


# In[ ]:




