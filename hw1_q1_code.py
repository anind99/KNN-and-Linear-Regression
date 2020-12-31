#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# Q1

# In[2]:


filereal = "clean_real.txt"
filefake = "clean_fake.txt"
metric = 'cosine'


# In[3]:


def load_data():
    with open(filereal, "r") as real:
        rlines = real.readlines()
    with open(filefake, "r") as fake:
        flines = fake.readlines()
    rlines = [x.strip() for x in rlines]
    flines = [x.strip() for x in flines]
    
    totalX = totalX= rlines + flines
    totalY = [1 for x in rlines] + [0 for x in flines]
    Y = np.array(totalY)
    
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(totalX)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.3, random_state=42)
    
    X_test, X_val, y_test, y_val = train_test_split(
        X_test, y_test, test_size=0.5, random_state=42)
    return X_train, X_val, X_test, y_train, y_val, y_test


# In[4]:


def select_knn_model(X_train, X_val, X_test, y_train, y_val, y_test, m="minkowski"):
    train_accuracy = np.empty(20)
    val_accuracy = np.empty(20)
    for i in range(20):
        neigh = KNeighborsClassifier(n_neighbors=i+1, metric= m)
        neigh.fit(X_train, y_train)
        train_accuracy[i] = neigh.score(X_train, y_train)
        
        val_accuracy[i] = neigh.score(X_val, y_val)
        
    plt.plot([i+1 for i in range(20)], train_accuracy, label = "training accuracy")
    plt.plot([i+1 for i in range(20)], val_accuracy, label = "validation accuracy")
    plt.xlabel("K")
    plt.ylabel("accuracy")
    plt.legend(loc="upper right")
    plt.show()

    
    k= 0
    m = max(val_accuracy)
    for i in range(len(val_accuracy)):
        if (val_accuracy[i] == m):
            k = i+1
    
    return k


# In[5]:


X_train, X_val, X_test, y_train, y_val, y_test = load_data()
print("Cosine Loss")
k = select_knn_model(X_train, X_val, X_test, y_train, y_val, y_test, metric)
print("Squared Loss")
k2 = select_knn_model(X_train, X_val, X_test, y_train, y_val, y_test)


# In[6]:


print("Cosine loss k is "+str(k))
neigh = KNeighborsClassifier(n_neighbors=k, metric= metric)
neigh.fit(X_train, y_train)
print("Accuracy: "+str(neigh.score(X_test, y_test)))
print("Squared Loss k is "+str(k2))
neigh = KNeighborsClassifier(n_neighbors=k)
neigh.fit(X_train, y_train)
print("Accuracy: "+str(neigh.score(X_test, y_test)))


# In[ ]:





# In[ ]:




