#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Load DataSet

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-lenght', 'sepal-width', 'petal-lenght', 'petal-width', 'class']
dataset = pandas.read_csv(url, names=names)


# In[2]:


# Shape
print(dataset.shape)


# In[3]:


# Head
print(dataset.head(20))


# In[4]:


# Description
print(dataset.describe())


# In[5]:


# Class distribution
print(dataset.groupby('class').size())


# In[6]:


# Import the necessary packages and modules
import matplotlib.pyplot as plt
import numpy as np

# Prepare Data
x = np.linspace(0,10,100)

# Plot Data
plt.plot(x, x, label = 'linear')

# Add legend
plt.legend()

# Show the plot
plt.show()


# In[7]:


import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot([1,2,3,4], [10,20,25,30], color = 'lightblue', linewidth = 3)
ax.scatter([0.3, 3.8, 1.2, 2.5], [11, 25, 9, 26], color = 'darkgreen', marker = '^')
ax.set_xlim(0.5, 4.5)
plt.show()


# In[8]:


# Box and whisker plots
dataset.plot(kind = 'box', subplots = True, layout = (2,2), sharex = False, sharey = False)
plt.show()


# In[9]:


# Histograms
dataset.hist()
plt.show()


# In[10]:


# Scatter plot matrix
scatter_matrix(dataset)
plt.show()


# In[ ]:




