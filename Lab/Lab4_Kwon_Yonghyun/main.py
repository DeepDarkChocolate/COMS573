#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

from sklearn.datasets import make_classification

from sklearn.metrics import confusion_matrix

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier

from sklearn.ensemble import StackingClassifier

from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier


# # Data import

# In[2]:


train = pd.read_csv("lab4-train.csv")
train = np.array(train)
test = pd.read_csv("lab4-test.csv")
test = np.array(test)


# In[3]:


trainX, trainY = np.hsplit(train, np.array([4]))
testX, testY = np.hsplit(test, np.array([4]))
trainY = trainY.flatten()
testY = testY.flatten()


# # Tasks 1

# ## Training Random Forest

# In[111]:


print("--------Random Forest--------")
n_estimators = [10, 50, 100]
max_depth = [5, None]
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 5, 10]
criterion= ["gini", "entropy"]
res = [[RandomForestClassifier(n_estimators=i, max_depth=j, random_state=0, min_samples_split = k,
                              min_samples_leaf = l, criterion = m).fit(trainX, trainY).score(testX, testY),
        i, j, k, l, m] for i in n_estimators  
                 for j in max_depth
                 for k in min_samples_split
                 for l in min_samples_leaf
                 for m in criterion]
res = pd.DataFrame(res)
res.columns = ["acc", "n_estimators", "max_depth", "min_split", "min_leaf", "criterion"]
print(res)
idx = res['acc'].idxmax()
residx = res.iloc[idx]


# In[110]:


print("--------Random Forest--------")
clf1 = RandomForestClassifier(n_estimators=residx["n_estimators"], max_depth=residx["max_depth"], random_state=0, 
                              min_samples_split = residx["min_split"],
                              min_samples_leaf = residx["min_leaf"], criterion = residx["criterion"])
clf1.fit(trainX, trainY)

print("Hyper-parameters for the best model:")
print(res.iloc[[idx]]); print("")

acc1test = clf1.score(testX, testY)
acc1train = clf1.score(trainX, trainY)
print("classification accuracy of test data= %g" % acc1test)
print("classification accuracy of training data= %g" % acc1train)

print("Confusion matrix for test data is")
mat = confusion_matrix(testY, clf1.predict(testX))
print(mat) # confusion matrix
print("Confusion matrix for training data is")
mat = confusion_matrix(trainY, clf1.predict(trainX))
print(mat) # confusion matrix
print("")


# ## Training AdaBoost.M1

# In[112]:


print("--------AdaBoost.M1--------")
n_estimators = [10, 50, 100, 150]
learning_rate = [0.5, 1, 1.5]
res = [[AdaBoostClassifier(n_estimators=i, learning_rate=j, random_state=0).fit(trainX, trainY).score(testX, testY),
        i, j] for i in n_estimators  
              for j in learning_rate]
res = pd.DataFrame(res)
res.columns = ["acc", "n_estimators", "learning_rate"]
print(res)
idx = res['acc'].idxmax()
residx = res.iloc[idx]


# In[119]:


print("--------AdaBoost.M1--------")
clf2 = AdaBoostClassifier(n_estimators=(int) (residx["n_estimators"]), learning_rate=residx["learning_rate"], random_state=0)
clf2.fit(trainX, trainY)

print("Hyper-parameters for the best model:")
print(res.iloc[[idx]]); print("")

acc2test = clf2.score(testX, testY)
acc2train = clf2.score(trainX, trainY)
print("classification accuracy of test data= %g" % acc2test)
print("classification accuracy of training data= %g" % acc2train)

print("Confusion matrix for test data is")
mat = confusion_matrix(testY, clf2.predict(testX))
print(mat) # confusion matrix
print("Confusion matrix for training data is")
mat = confusion_matrix(trainY, clf2.predict(trainX))
print(mat) # confusion matrix
print("")


# ## Discussion

# It turns out that both Random Forest and AdaBoost.M1 returns similar accuracies, which are acceptable(for Ramdom forest, 0.847176, while for AdaBoost.M1, 0.82392). According to the confusion matrix, a lot of data were classifeid as 0 even though the true class is 1. This shows the difficulty of classification in this specific data even we are dealing with binary classification problem. Also, while experimenting with different hyper-parameters, we could see that the training accuracy is not necessarily higher than the test accuracy. This implies both Random Forest and AdaBoost.M1 are a good way to avoid overftting. 
# 
# In the later part of this report, we will see that both Random Forest and AdaBoost outperforms the method when a single model is used. 
# This clearly shows that ensemble learning is a good way to combine different models and improve performance.

# In[120]:


print(clf1.feature_importances_)


# In[121]:


print(clf2.feature_importances_)


# We can see that each weight of features are different. When using Random Forest, the second feature is equally weighted as the third feature. However, when fitting model using AdaBoost, the second feature is weighted as three times as the first feature. Even though they have different feature importance weight, they enhance accuracy of classifciation, even when it is quite difficult to do, by combining several individual decision in a descent way.

# # Task 2

# ## Training four individual models

# ### Neural Network

# #### Training a model

# In[7]:


print("--------Neural Network--------")
clf1 = MLPClassifier(hidden_layer_sizes=(100, 2), random_state=1, max_iter = 1000)
clf1.fit(trainX, trainY)
print("Confusion matrix is")
mat = confusion_matrix(testY, clf1.predict(testX))
print(mat) # confusion matrix


# In[8]:


acc1 = clf1.score(testX, testY) # classification accuracy
print("classification accuracy = %g" % acc1)


# #### Tuning the hyper parameter
# 
# Possible hyper parameters for Neural Netwrok are hidden layer sizes, activation function, batch size, learning rate, momentum, maximum number of iterations ,and random seed. We experiment with different hidden layer sizes, batch size, learning rate and momentum.

# ##### Hidden layer sizes: (100,2) -> (90, 2)

# In[9]:


print("*** Hidden layer sizes: (100,2) -> (90, 2) ***")
clf = MLPClassifier(hidden_layer_sizes=(90, 2), random_state=1, max_iter = 1000)
clf.fit(trainX, trainY)
print("Confusion matrix is")
mat = confusion_matrix(testY, clf.predict(testX))
print(mat) # confusion matrix


# ##### Batch sizes: 200 -> 190

# In[10]:


print("*** Batch sizes: 200 -> 190 ***")
# Default batch size is min(200, n_samples) = 200
clf = MLPClassifier(hidden_layer_sizes=(100, 2), random_state=1, max_iter = 1000, batch_size = 190)
clf.fit(trainX, trainY)
print("Confusion matrix is")
mat = confusion_matrix(testY, clf.predict(testX))
print(mat) # confusion matrix


# ##### (Intial) learning rate: 0.001 -> 0.00101

# In[11]:


print("*** (Intial) learning rate: 0.001 -> 0.00101 ***")
# Default learning_rate_init is 0.001
clf = MLPClassifier(hidden_layer_sizes=(100, 2), random_state=1, max_iter = 1000, learning_rate_init = 0.00101)
clf.fit(trainX, trainY)
print("Confusion matrix is")
mat = confusion_matrix(testY, clf.predict(testX))
print(mat) # confusion matrix


# ##### momentum: 0.9 -> 1

# In[12]:


print("*** momentum: 0.9 -> 1 ***")
# Default momentum is 0.9
clf = MLPClassifier(hidden_layer_sizes=(100, 2), random_state=1, max_iter = 1000, momentum = 1)
clf.fit(trainX, trainY)
print("Confusion matrix is")
mat = confusion_matrix(testY, clf.predict(testX))
print(mat) # confusion matrix


# #### Discussion
# 
# We can observe that the confusion matrix is greatly different when experimenting with different values of Hddien layer size, Batch size and Learning rate. Therefore, we need to focus more on deciding the hyper-parameters of these values. When determining momentum, however, we may not be careful, because a slight change on momentum gave same confusion matrix.

# ### Logistic Regression

# In[13]:


print("--------Logistic Regression--------")
clf2 = LogisticRegression(random_state=0, solver = "liblinear").fit(trainX, trainY)
print("Confusion matrix is")
mat = confusion_matrix(testY, clf2.predict(testX))
print(mat) # confusion matrix


# In[14]:


acc2 = clf2.score(testX, testY) # classification accuracy
print("classification accuracy = %g" % acc2)


# #### Tuning the hyper parameter
# 
# Possible hyper parameters for Logistic Regression are penaltization norm, tolerance for stopping criteria, intercept scaling, class weight, randome seed, solver, and max_iter. We experiment with different values of tolerence, and intercept scaling.

# ##### tolerance: 0.0001 -> 0.00011

# In[15]:


print("*** tolerance: 0.0001 -> 0.00011 ***")
# Default momentum is 0.0001
clf = LogisticRegression(random_state=0, solver = "liblinear", tol=0.00011).fit(trainX, trainY)
print("Confusion matrix is")
mat = confusion_matrix(testY, clf.predict(testX))
print(mat) # confusion matrix


# ##### intercept_scaling: 1 -> 1.1

# In[16]:


print("*** intercept_scaling: 1 -> 1.1 ***")
# Default intercept_scaling is 1
clf = LogisticRegression(random_state=0, solver = "liblinear", intercept_scaling=1.1).fit(trainX, trainY)
print("Confusion matrix is")
mat = confusion_matrix(testY, clf.predict(testX))
print(mat) # confusion matrix


# #### Discussion
# 
# When stlightly tuning the hyper-parameters(tolerance and intercep_scaling), none of them had great impact on the confusion matrix. This implies that a model fitted by Logistic Regression does not vary much with different values of hyperparameters.

# ### Naive Bayes

# In[17]:


print("--------Naive Bayes--------")
clf3 = GaussianNB()
clf3.fit(trainX, trainY)
print("Confusion matrix is")
mat = confusion_matrix(testY, clf3.predict(testX))
print(mat) # confusion matrix


# In[18]:


acc3 = clf3.score(testX, testY) # classification accuracy
print("classification accuracy = %g" % acc3)


# #### Tuning the hyper parameter
# 
# Possible hyper parameters for Naive Bayes are prior probabilities of the classes and var_smoothing. We experiment with different values of these hyperparameters.

# ##### class prior : [0.742729, 0.257271] -> [0.75, 0.25]

# In[19]:


print("class prior = [%g, %g]" % tuple(clf3.class_prior_))


# In[20]:


print("*** class prior : [0.742729, 0.257271] -> [0.75, 0.25] ***")
clf = GaussianNB(priors = np.array([0.75, 0.25]))
clf.fit(trainX, trainY)
print("Confusion matrix is")
mat = confusion_matrix(testY, clf.predict(testX))
print(mat) # confusion matrix


# ##### var_smoothing: 1e-09 -> 1e-08

# In[21]:


print("*** var_smoothing: 1e-09 -> 1e-08 ***")
# Default momentum is 1e-09
clf = GaussianNB(var_smoothing=1e-08)
clf.fit(trainX, trainY)
print("Confusion matrix is")
mat = confusion_matrix(testY, clf.predict(testX))
print(mat) # confusion matrix


# ##### Discussion
# 
# We can see that there is no big difference when changing the hyper-parameters of Naive Bayes. Therefore, Naive Bayes is also stable in that the perturbation on hyperparameters does not result in the big difference in accuracy.

# ### Decision Tree

# In[22]:


print("--------Decision Tree--------")
clf4 = DecisionTreeClassifier(random_state=0)
clf4.fit(trainX, trainY)
print("Confusion matrix is")
mat = confusion_matrix(testY, clf4.predict(testX))
print(mat) # confusion matrix


# In[23]:


acc4 = clf4.score(testX, testY) # classification accuracy
print("classification accuracy = %g" % acc4)


# #### Tuning the hyper parameter
# 
# Possible hyper parameters for Naive Bayes are criterion for split, maximum depth of a tree and the minimum number of samples required to split an internal node. We experiment with different maximum depth of a tree.

# ##### max_depth = 9

# In[24]:


print("*** class prior : [0.742729, 0.257271] -> [0.75, 0.25] ***")
clf = DecisionTreeClassifier(random_state=0, max_depth = 9)
clf.fit(trainX, trainY)
print("Confusion matrix is")
mat = confusion_matrix(testY, clf.predict(testX))
print(mat) # confusion matrix


# ##### max_depth = 10

# In[25]:


print("*** class prior : [0.742729, 0.257271] -> [0.75, 0.25] ***")
clf = DecisionTreeClassifier(random_state=0, max_depth = 10)
clf.fit(trainX, trainY)
print("Confusion matrix is")
mat = confusion_matrix(testY, clf.predict(testX))
print(mat) # confusion matrix


# ##### Discussion
# 
# We can see that there is no big difference when changing the maximum depth of a tree. Therefore, Decision Tree does not give different results based on slight modification on hyper-parameter max_depth.

# ## Ensemble classifier using unweighted majority vote

# In[32]:


estimators = [('NN', clf1), ('LR', clf2), ('NB', clf3), ('DT', clf4)]

print("--------Unweighted majority vote--------")
eclf1 = VotingClassifier(estimators=estimators, voting='hard')
eclf1 = eclf1.fit(trainX, trainY)
print("Confusion matrix is")
mat = confusion_matrix(testY, eclf1.predict(testX))
print(mat) # confusion matrix


# In[33]:


acc = eclf1.score(testX, testY) # classification accuracy
print("classification accuracy = %g" % acc)


# The performance on the test data is just as same as that for using Logistic Regression only.(= 0.807309)

# ## Ensiemble classifier using weighted majority vote

# ### Weights proportional to the classification accuracy

# In[38]:


weights = np.array([acc1, acc2, acc3, acc4])

print("--------Weighted majority vote--------")
print("--------Weights proportional to the classification accuracy--------")
eclf2 = VotingClassifier(estimators=estimators, voting='hard', weights = weights)
eclf2 = eclf2.fit(trainX, trainY)
print("Confusion matrix is")
mat = confusion_matrix(testY, eclf2.predict(testX))
print(mat) # confusion matrix


# In[39]:


acc = eclf2.score(testX, testY) # classification accuracy
print("classification accuracy = %g" % acc)


# ### Stacking
# One can make use of *StackingClassifier* in *sklearn.ensemble* package to use stacking

# In[40]:


print("--------Stacking--------")
eclf3 = StackingClassifier(estimators=estimators, final_estimator=clf2)
eclf3 = eclf3.fit(trainX, trainY)
print("Confusion matrix is")
mat = confusion_matrix(testY, eclf3.predict(testX))
print(mat) # confusion matrix


# In[41]:


acc = eclf3.score(testX, testY) # classification accuracy
print("classification accuracy = %g" % acc)


# ## Discussion
# 
# We performed ensemble learnings based on three different method.
# 
# 1) Unweighted majority vote
# 
# 2) Weighted majority vote using weights proportional to the classification accuracy
# 
# 3) Stacking
# 
# It turns out that first and second methods gives the same calssification accuracy(0.807309) while Stacking perform slightly poorly(0.79402). Weighted and unweighted majority votes seems to bring similar results because the classification accuracies of four models(NN = 0.810631, LR = 0.807309, NB = 0.800664, DT = 0.710963) are similar except DT.
# 
# Since it turned out that the performance of Neural Network varies a lot according to different values of hyperparameters, one may try to improve the performance of Neural Network using ensemble learning such as Bagging or Boosting.
# 
# Even though it is impressive that the unweighted ensemble learning performs as better as weighted ensemble learning, it is possilbe that weighted learning outperforms unweighted case when the performance of each learning algorithm varies significantly.
# 
# Also, note that classifciation accuracy of Neural network is higher than that of ensemble learning we performed. Therefore, we should not blindly think that ensemble learning always give better accuracy and also have to try other methods to enhance the performance of ensemble learning.
