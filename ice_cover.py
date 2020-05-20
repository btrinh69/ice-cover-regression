#!/usr/bin/env python
# coding: utf-8

# In[1]:


import csv

def get_dataset():
    file = open("IceDays.csv")
    reader = csv.reader(file)
    dataset = list(reader)
    file.close()
    dataset = dataset[1:]
    for i in dataset:
        i[0] = int(i[0][:4])
        i[1] = int(i[1])
    return dataset


# In[2]:


def get_dataset_normalized():
    data = get_dataset()
    mean = 0
    for i in data:
        mean += i[0]
    mean /= len(data)
    std = 0
    for i in data:
        std += (i[0]-mean)**2
    std = (std/(len(data)-1))**0.5
    for i in range(len(data)):
        data[i][0] = (data[i][0]-mean)/std
    return data


# In[3]:


import numpy as np

def print_stats(data):
    print(len(data))
    mean = 0
    for i in data:
        mean += i[1]
    mean /= len(data)
    print('{:.2f}'.format(mean))
    dev = 0
    for i in data:
        dev += (i[1]-mean)**2
    dev = (dev/(len(data)-1))**0.5
    print('{:.2f}'.format(dev))


# In[5]:


def regression(b0, b):
    sum_ = 0
    data = get_dataset()
    for i in data:
        sum_ += (b0+b*i[0]-i[1])**2
    sum_ /= len(data)
    return sum_


# In[11]:


def regression_normalized(b0, b):
    sum_ = 0
    data = get_dataset_normalized()
    for i in data:
        sum_ += (b0+b*i[0]-i[1])**2
    sum_ /= len(data)
    return sum_


# In[12]:


def gradient_descent(b0, b):
    sum0 = 0
    sum1 = 0
    data = get_dataset()
    for i in data:
        sum0 += b0 + b*i[0] - i[1]
        sum1 += (b0 + b*i[0] - i[1])*i[0]
    sum0 = sum0*2/len(data)
    sum1 = sum1*2/len(data)
    return (sum0, sum1)


# In[18]:


def gradient_descent_normalized(b0, b):
    sum0 = 0
    sum1 = 0
    data = get_dataset_normalized()
    for i in data:
        sum0 += b0 + b*i[0] - i[1]
        sum1 += (b0 + b*i[0] - i[1])*i[0]
    sum0 = sum0*2/len(data)
    sum1 = sum1*2/len(data)
    return (sum0, sum1)


# In[19]:


import random as rd

def gradient_descent_stochastic(b0, b):
    data = get_dataset_normalized()
    i = data[rd.randint(0, len(data)-1)]
    sum0 = (b0 + b*i[0] - i[1])*2
    sum1 = (b0 + b*i[0] - i[1])*i[0]*2
    return (sum0, sum1)


# In[20]:


def iterate_gradient(T, eta):
    beta0 = 0
    beta1 = 0
    for i in range(1, T+1):
        (b0, b1) = gradient_descent(beta0, beta1)
        beta0 -= eta*b0
        beta1 -= eta*b1
        print('{} {:.2f} {:.2f} {:.2f}'.format(i, beta0, beta1, regression(beta0, beta1)))


# In[25]:


def compute_betas():
    data = get_dataset()
    meanX = 0
    meanY = 0
    for i in data:
        meanX += i[0]
        meanY += i[1]
    meanX /= len(data)
    meanY /= len(data)
    b1 = 0
    dev = 0
    b0 = 0
    for i in data:
        b1 += (i[0]-meanX)*(i[1]-meanY)
        dev += (i[0]-meanX)**2
    b1 /= dev
    b0 = meanY - b1*meanX
    return (b0, b1, regression(b0, b1))


# In[30]:


def predict(year):
    (b0, b1, mse) = compute_betas()
    return (b0+b1*year)


# In[28]:


def iterate_normalized(T,eta):
    beta0 = 0
    beta1 = 0
    for i in range(1, T+1):
        (b0, b1) = gradient_descent_normalized(beta0, beta1)
        beta0 -= eta*b0
        beta1 -= eta*b1
        print('{} {:.2f} {:.2f} {:.2f}'.format(i, beta0, beta1, regression_normalized(beta0, beta1)))


# In[32]:


def sgd(T, eta):
    data = get_dataset_normalized()
    beta0 = 0
    beta1 = 0
    for i in range(1, T+1):
        (b0, b1) = gradient_descent_stochastic(beta0, beta1)
        beta0 -= eta*b0
        beta1 -= eta*b1
        print('{} {:.2f} {:.2f} {:.2f}'.format(i, beta0, beta1, regression_normalized(beta0, beta1)))

