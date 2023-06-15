#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[2]:


import os 
path = os.listdir(r'C:\Users\USER\Desktop\BTD/')
classes = {'no tumor':0, 'pituitary':1}


# In[3]:


import cv2
X = []
Y = []
for cls in classes:
    pth = r'C:\Users\USER\Desktop\BTD/'+cls
    for j in os.listdir(pth):
        img = cv2.imread(pth+'/'+j, 0)
        img = cv2.resize(img, (200,200))
        X.append(img)
        Y.append(classes[cls])


# In[4]:


np.unique(Y)


# In[5]:


X = np.array(X)
Y = np.array(Y)


# In[7]:


pd.Series(Y).value_counts()


# In[8]:


X.shape


# In[9]:


plt.imshow(X[0], cmap='gray')


# In[10]:


X_updated = X.reshape(len(X), -1)
X_updated.shape


# In[11]:


xtrain, xtest, ytrain, ytest = train_test_split(X_updated, Y,random_state=10, test_size=.20)


# In[12]:


xtrain.shape, xtest.shape


# In[15]:


print(xtrain.max(), xtrain.min())
print(xtest.max(), xtest.min())
xtrain = xtrain/255
xtest = xtest/255
print(xtrain.max(), xtrain.min())
print(xtest.max(), xtest.min())


# In[16]:


from sklearn.decomposition import PCA


# In[17]:


print(xtrain.shape, xtest.shape)

pca = PCA(.98)
#pca_train = pca.fit_transform(xtrain)
#pca_test =  pca.transform(xtest)
pca_train = xtrain
pca_test = xtest


# In[18]:


#print(pca_train.shap, pca_test.shape)
#print(pca.n_components_)
#print(pca.n_features_)


# In[19]:


from sklearn.linear_model  import LogisticRegression
from sklearn.svm import SVC


# In[22]:


lg = LogisticRegression(C=0.1)
lg.fit(pca_train, ytrain)


# In[23]:


sv = SVC()
sv.fit(pca_train, ytrain)


# In[24]:


print("Training Score:", lg.score(pca_train, ytrain))
print("Testing Score:", sv.score(pca_test, ytest))


# In[25]:


print("Training Score:", sv.score(pca_train, ytrain))
print("Testing Score:", sv.score(pca_test, ytest))


# In[26]:


pred = sv.predict(pca_test)
np.where(ytest!=pred)


# In[35]:


pred[6]


# In[36]:


ytest[6]


# In[46]:


dec = {0: 'No Tumor', 1:'Positive Tumor'}


# In[47]:


plt.figure(figsize=(12,8))
p = os.listdir(r'C:\Users\USER\Desktop\BTD/')
c = 1
for i in os.listdir(r'C:\Users\USER\Desktop\BTD/no tumor/')[:9]:
    plt.subplot(3,3,c)
    
    img = cv2.imread(r'C:\Users\USER\Desktop\BTD/no tumor/'+i,0)
    img1 = cv2.resize(img, (200,200))
    img1 = img1.reshape(1,-1)/255
    p = sv.predict(img1)
    plt.title(dec[p[0]])
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    c+=1


# In[48]:


plt.figure(figsize=(12,8))
p = os.listdir(r'C:\Users\USER\Desktop\BTD/')
c = 1
for i in os.listdir(r'C:\Users\USER\Desktop\BTD/pituitary/')[:16]:
    plt.subplot(4,4,c)
    
    img = cv2.imread(r'C:\Users\USER\Desktop\BTD/pituitary/'+i,0)
    img1 = cv2.resize(img, (200,200))
    img1 = img1.reshape(1,-1)/255
    p = sv.predict(img1)
    plt.title(dec[p[0]])
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    c+=1


# In[ ]:




