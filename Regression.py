
# coding: utf-8

# # Initialization

# In[17]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
train=pd.read_csv("C:\\Users\\dell\\Desktop\\spyder\\train.csv")
test=pd.read_csv("C:\\Users\\dell\\Desktop\\spyder\\test.csv")


# # Calculate Error

# In[4]:


def Jtheta(x,y,theta):
    y_pred=x.dot(theta)
    c=sum([(np.round(val,2)**2) for val in (y_pred-y)])
    #t_theta=sum([v**2 for v in theta])
    cost=c/(2*len(x))
    return cost


# # Gradient Descent

# In[5]:


def gradient_descent(x,y,theta,alpha,lemda):
    y_pred=x.dot(theta)
    c=y_pred-y
    grad = (x.T.dot(c))/len(x)
    temp = theta-(alpha*grad)
    return temp


# # Predict Function

# In[6]:


def predict(theta,test):
    x0_test = np.ones((len(test),1))
    test.insert(loc = 0,column='x0',value=x0_test)
    pred =test.dot(theta)
    return pred    


# # Iterate and minimize cost

# In[20]:


theta=np.array([1,1,1,1,1])
iteration=100
alpha=0.000001
lemda=0.000001
elist=[]
m=train.shape[1]
x = train.iloc[:,0:4]
y = train.iloc[:,4]
x0 = np.ones((len(x),1))
x.insert(loc = 0,column='x0',value=x0)
for i in range(iteration):
    error=Jtheta(x,y,theta)
    if error<0.00001:
        break
    else:
        elist.append(error)
        theta=gradient_descent(x,y,theta,alpha,lemda)

#plt.plot(list(range(100)),elist)
#plt.xlabel('no. of iteration ')
#plt.ylabel('cost')
#plt.show()


# In[21]:


theta


# In[22]:


pred=predict(theta,test)


# In[23]:


pred


# In[ ]:


from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

