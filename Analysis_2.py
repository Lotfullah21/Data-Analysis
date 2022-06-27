#!/usr/bin/env python
# coding: utf-8

# In[30]:


import numpy as np
data = np.load('X.npy')
data.shape


# In[31]:


data


# In[10]:


data.max()


# In[32]:


print(np.max(data[:]))


# In[33]:


np.where(data == data.max())


# In[34]:


print(data[:,32023].max())


# ## Transformation

# In[7]:


# Transformation 
t_data = np.log2(data+1)
t_data.shape


# In[9]:


t_data.max()


# In[14]:


print(np.max(data[:,0]))


# In[17]:


print(np.max(t_data[:32]))


# In[36]:


np.where(t_data == t_data.max())


# In[ ]:


# as we can see the maximum element is in row 66 and column 32023


# In[35]:


np.where(t_data == t_data.min())
# we have multiple minimum data points


# In[37]:


print(t_data[:,32023].max())


# # PCA

# In[38]:


from sklearn.decomposition import PCA


# In[58]:


pca = PCA()


# In[ ]:





# In[42]:


plt.plot(np.arange(0,511),np.cumsum(pca.explained_variance_ratio_))
plt.title("Cumulative Variance Explained",size=18)
plt.xlabel("Number of Components",size=14)
plt.ylabel("% Variance Explained",size=14)
plt.show()


# In[43]:


# Number of PCs needed for raw data
np.where(np.cumsum(pca.explained_variance_ratio_) >=.85)[0][0]


# In[45]:


pca2 = PCA(n_components = 50)

pca2.fit(t_data)

da = pca2.explained_variance_ratio_
print(da)


# In[50]:


pca1 = PCA(n_components = 50)

pca1.fit(data)

va = pca1.explained_variance_ratio_
print(va)


# In[53]:


pca = PCA().fit(t_data)
pcs = pca.transform(t_data)
plt.scatter(pcs[:,0],pcs[:,1],c=pca.explained_variance_ratio_)
plt.title("Wine Data PCs",size=18)
plt.xlabel("PC 1",size=14)
plt.ylabel("PC 2",size=14)
plt.axis("equal")
plt.show()


# In[56]:


from sklearn.manifold import MDS


# In[59]:


mds = MDS(n_components=50,verbose=1,eps=1e-5)
mds.fit(t_data)
plt.scatter(mds.embedding_[:,0],mds.embedding_[:,1])
plt.title("MDS Plot",size=18)
plt.axis("equal")
plt.show()


# In[60]:


from sklearn.manifold import TSNE


# In[62]:


tsne = TSNE(n_components=3,verbose=1,perplexity=40)
z_tsne = tsne.fit_transform(t_data)
plt.scatter(z_tsne[:,0],z_tsne[:,1])
plt.title("TSNE, perplexity 40",size=18)
plt.axis("equal")
plt.show()


# In[65]:


from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=4,n_init=10)
y = kmeans.fit_predict(t_data)
plt.scatter(t_data[:,0],t_data[:,1],c=y)
plt.title("KMeans Clustering, PCA Plot",size=18)
plt.xlabel("PC 1",size=14)
plt.ylabel("PC 2",size=14)
plt.axis("equal")
plt.show()


# In[66]:


plt.scatter(mds.embedding_[:,0],mds.embedding_[:,1],c=y)
plt.title("KMeans Clustering, MDS Plot",size=18)
plt.axis("equal")
plt.show()


# In[67]:


plt.scatter(z_tsne[:,0],z_tsne[:,1],c=y)
plt.title("KMeans Clustering, TSNE Plot",size=18)
plt.axis("equal")
plt.show()


# In[ ]:


all_kmeans = [KMeans(n_clusters=i+1,n_init=10) for i in range(8)]
# i-th kmeans fits i+1 clusters
for i in range(8):
    all_kmeans[i].fit(t_data)

inertias = [all_kmeans[i].inertia_ for i in range(8)]
plt.plot(np.arange(1,9),inertias)
plt.title("KMeans Sum of Squares Criterion",size=18)
plt.xlabel("# Clusters",size=14)
plt.ylabel("Within-Cluster Sum of Squares",size=14)
plt.show()


# In[ ]:




