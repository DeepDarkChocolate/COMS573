#!/usr/bin/env python
# coding: utf-8

# In[1]:
import numpy as np
import pandas as pd

import arff

#sudo pip install python-weka-wrapper3
#sudo pip install javabridge
#Go to https://fracpete.github.io/python-weka-wrapper/install.html for more information

import weka.core.jvm as jvm
from weka.core.dataset import Instances
import weka.core.converters as converters
from weka.classifiers import Classifier, Evaluation
from weka.core.classes import Random
import weka.plot.graph as graph

jvm.start()

print("Numpy version = %s" % np.__version__)
print("Pandas version = %s" % pd.__version__)
print("arff version = %s" % arff.__version__)
print("python-weka-wrapper3 version = %s" % "0.1.12")
print("javabridge version = %s" % "1.0.18")
# In[2]:


# inport data
data_dir = "https://archive.ics.uci.edu/ml/machine-learning-databases/voting-records/"
data1 = pd.read_csv(data_dir + "house-votes-84.data", header=None)
data1 = data1.reindex(columns=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,0])
data1 = np.array(data1)

# create arff file
obj = {
   'description': u'',
   "relation": "vote",
   'attributes': [
       ("handicapped-infants", ["n", "y"]),
       ('water-project-cost-sharing', ['n', 'y']),
       ('adoption-of-the-budget-resolution', ['n', 'y']),
       ('physician-fee-freeze', ['n', 'y']),
       
       ('el-salvador-aid', ['n', 'y']),
       ('religious-groups-in-schools', ['n', 'y']),
       ('anti-satellite-test-ban', ['n', 'y']),
       ('aid-to-nicaraguan-contras', ['n', 'y']),
       
       ('mx-missile', ['n', 'y']),
       ('immigration', ['n', 'y']),
       ('synfuels-corporation-cutback', ['n', 'y']),
       ('education-spending', ['n', 'y']),
       
       ('superfund-right-to-sue', ['n', 'y']),
       ('crime', ['n', 'y']),
       ('duty-free-exports', ['n', 'y']),
       ('export-administration-act-south-africa', ['n', 'y']),
       
       ('\'Class\'', ['democrat', 'republican']),
   ],
   'data': data1,
}
fp = open("vote2.arff", "w")
arff.dump(obj, fp)
fp.close()

# load data
data = converters.load_any_file("vote2.arff")
data.class_is_last()


# In[3]:


cls = Classifier(classname="weka.classifiers.trees.J48")
cls.build_classifier(data)
#print(cls.to_help())
print(cls)


# In[12]:


graph.plot_dot_graph(cls.graph, "Tree.png")


# ![Tree](Tree.png)

# In[5]:


n = 5
evaluation = Evaluation(data)                     # initialize with priors
evaluation.crossvalidate_model(cls, data, n, Random(1))  # 5-fold CV
print("Accuracy = %g" % evaluation.percent_correct + "%")

z = 1.96
accuracy = evaluation.percent_correct/100
margin = z * np.sqrt( (accuracy * (1 - accuracy)) / n)
print("95% "+"Confidence Interval = (%g, %g)" % (accuracy - margin, accuracy + margin))

print(evaluation.summary())

#print("Number of incorrect = %g" % evaluation.incorrect)
print(evaluation.class_details())


# In[6]:


n = 5
seed = 1
rnd = Random(seed)
rand_data = Instances.copy_instances(data)
rand_data.randomize(rnd)
classifier = Classifier(classname="weka.classifiers.trees.J48")

for i in range(n):
    train = rand_data.train_cv(n, i)
    test = rand_data.test_cv(n, i)

    cls = Classifier.make_copy(classifier)
    cls.build_classifier(train)
    evaluation = Evaluation(rand_data)
    evaluation.test_model(cls, train)

    print("-------------%g-th fold-------------" % i)
    print("Accuracy for training data = %g" % evaluation.percent_correct + "%")
    
    evaluation = Evaluation(rand_data)
    evaluation.test_model(cls, test)
    print("Accuracy for test data = %g" % evaluation.percent_correct + "%")
    
    graph.plot_dot_graph(cls.graph, ("Tree" + str(i) + ".png"))

jvm.stop()