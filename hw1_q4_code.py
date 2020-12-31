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


# In[2]:


data_train = {'X': np.genfromtxt('data_train_X.csv', delimiter = ','), 
                                 't': np.genfromtxt('data_train_y.csv', delimiter = ',')}


# In[3]:


data_test = {'X': np.genfromtxt('data_test_X.csv', delimiter = ','), 
                                 't': np.genfromtxt('data_test_y.csv', delimiter = ',')}


# In[4]:


train_data = (data_train['t'], data_train['X'])
test_data = (data_test['t'], data_test['X'])


# In[5]:


def shuffle_data(data):
    a,b = data[0], data[1]
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)
    return (a,b)


# In[6]:


def split_data(data, num_folds, fold):
    size = data[0].shape[0]
    splitsize = size/num_folds
    data_rest0 = []
    data_rest1 = []
    partitioned_data = np.empty(num_folds, dtype = object)
    for i in range(num_folds - 1):
        partitioned_data[i] = (data[0][int(i*splitsize):int((i+1)*splitsize)], 
                              data[1][int(i*splitsize):int((i+1)*splitsize)])
        
    partitioned_data[num_folds-1] = (data[0][int((num_folds-1)*splitsize):], 
                              data[1][int((num_folds-1)*splitsize):])
    
    for i in range(num_folds-1):
        if (i!=fold-1):
            if (data_rest0==[]):
                data_rest0 = list(partitioned_data[i][0])
                data_rest1 = list(partitioned_data[i][1])
            else:
                data_rest0 += list(partitioned_data[i][0])
                data_rest1 += list(partitioned_data[i][1])
                
    data_rest = (np.array(data_rest0),np.array(data_rest1))
    return partitioned_data[fold-1], data_rest


# In[7]:


def train_model(data, lamda):
    xtx = np.dot(data[1].transpose(),data[1])
    xtt = np.dot(data[1].transpose(),data[0])
    nlamdaI = lamda* np.identity(xtt.shape[0])
    
    return np.dot(np.linalg.inv(xtx+nlamdaI), xtt)


# In[8]:


def predict(data, model):
    return np.dot(model,data[1].transpose())


# In[9]:


def error(data,model):
    return (np.linalg.norm(predict(data,model)-data[0])**2)/(2*data[0].shape[0])


# In[10]:


def cross_validation(data, num_folds, lambd_seq):
    data = shuffle_data(data)
    cv_error = []
    for i in range(len(lambd_seq)):
        lambd = lambd_seq[i]
        cv_loss_lmd = 0
        for j in range(num_folds):
            val_cv, train_cv = split_data(data, num_folds, j)
            model = train_model(train_cv, lambd)
            cv_loss_lmd += error(val_cv, model)
        cv_error += [cv_loss_lmd/num_folds]
    return cv_error


# In[11]:


lambd_seq = [1,2,4,9,16,25,36,49,90]


# In[12]:


cv_error5 = cross_validation(train_data, 5, lambd_seq)
cv_error10 = cross_validation(train_data, 10, lambd_seq)


# In[13]:


training_error, testing_error = [], []
for num in lambd_seq:
    model = train_model(train_data, num)
    training_error += [error(train_data,model)]
    testing_error += [error(test_data,model)]


# In[14]:


print("training error: "+str(training_error))
print("testing error: "+str(testing_error))


# In[16]:


plt.plot(lambd_seq, training_error, label = "training")
plt.plot(lambd_seq, cv_error5, label = "5 fold cv")
plt.plot(lambd_seq, cv_error10,'--', label = "10 fold cv")
plt.plot(lambd_seq, testing_error, label = "testing")
plt.legend(loc="lower right")
plt.xlabel("N lambda")
plt.ylabel("Squared error")
plt.show()


# In[ ]:




