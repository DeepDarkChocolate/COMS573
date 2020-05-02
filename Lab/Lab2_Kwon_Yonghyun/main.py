from __future__ import absolute_import, division, print_function, unicode_literals
import functools

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from keras.utils import to_categorical

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import csv

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import pandas as pd

from timeit import default_timer as timer

import scipy.stats

print("TensorFlow version")
print(tf.__version__)
print("Keras version")
print(keras.__version__)

TRAIN_DATA_URL = "http://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tra"
TEST_DATA_URL = "http://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tes"

train_file_path = tf.keras.utils.get_file("optdigits.tra", TRAIN_DATA_URL)
test_file_path = tf.keras.utils.get_file("optdigits.tes", TEST_DATA_URL)

train_labels = []
train_images = []
with open(train_file_path, "r") as f:
    rdr = csv.reader(f)
    for line in rdr:
        train_labels.append(int(line[64]))
        train_images.append([int(i) for i in line[0:64]])

test_labels = []
test_images = []
with open(test_file_path, "r") as f:
    rdr = csv.reader(f)
    for line in rdr:
        test_labels.append(int(line[64]))
        test_images.append([int(i) for i in line[0:64]])
        
train_labels = np.array(train_labels)
train_images = np.array(train_images)
test_labels = np.array(test_labels)
test_images = np.array(test_images)
train_labels_full = train_labels
train_images_full = train_images

train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=0.20, shuffle= True)

train_cnn_full = train_images_full.reshape(train_images_full.shape[0], 8, 8, 1).astype('float32')
test_cnn = test_images.reshape(test_images.shape[0], 8, 8, 1).astype('float32')

train_cnn, val_cnn, train_cnn_labels, val_cnn_labels = train_test_split(train_cnn_full, train_labels_full, 
                                                                        test_size=0.20, shuffle= True)

encoded_train_labels = to_categorical(train_labels)
encoded_val_labels = to_categorical(val_labels)
encoded_test_labels = to_categorical(test_labels)
encoded_train_labels_full = to_categorical(train_labels_full)

encoded_train_labels_cnn = to_categorical(train_cnn_labels)
encoded_val_labels_cnn = to_categorical(val_cnn_labels)
encoded_test_labels_cnn = to_categorical(test_labels)
encoded_train_labels_full_cnn = to_categorical(train_labels_full)


def validation1(loss, activation, scale = 1, nHiddenlayers = 1, nHiddenunits = 100, lr = 0.01,
                momentum = 0.0, batch_size = 32, verbose = 0):
    #loss = "categorical_crossentropy"
    
    model = keras.Sequential()

    for i in range(int(nHiddenlayers)):
        model.add(keras.layers.Dense(nHiddenunits, activation=activation))

    model.add(keras.layers.Dense(10, activation='softmax'))
    
    model.compile(optimizer=keras.optimizers.SGD(lr = lr, momentum = momentum),
              loss= loss,
              metrics=['accuracy'])

    callbacks = [
        # Interrupt training if `val_loss` stops improving for over 2 epochs
        tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss')
        ]
    
    start = timer()
    modelFit = model.fit(train_images*scale, encoded_train_labels , epochs=20, callbacks=callbacks,
         validation_data=(val_images*scale, encoded_val_labels), verbose = verbose, batch_size = batch_size)
    end = timer()
    
    time = end - start
    num_iter = len(modelFit.history['loss'])
    
    train_acc = model.evaluate(train_images, encoded_train_labels, verbose=0)[1]
    val_acc = model.evaluate(val_images, encoded_val_labels, verbose=0)[1]
    
    return time, num_iter, train_acc, val_acc

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m-h,m, m+h, h

def performance(M, loss, activation, scale = [1], nHiddenlayers = [1], nHiddenunits = [100], 
                lr = [0.01], momentum = [0.0], batch_size = [32]):
    res = np.empty((0, 8))
    for i1 in scale:
        for i2 in nHiddenlayers:
            for i3 in nHiddenunits:
                for i4 in lr:
                    for i5 in momentum:
                        for i6 in batch_size:
                            res2 = np.empty((0, 4))
                            for i in range(M):
                                res2 = np.append(
                                    res2, np.array([
                                        validation1(scale = i1, nHiddenlayers = i2,
                                                   nHiddenunits = i3, lr = i4,
                                                   momentum = i5, batch_size = i6,
                                                   loss = loss, activation = activation)]), axis = 0)
                            res = np.append(res, np.array([np.reshape(np.apply_along_axis(mean_confidence_interval, 0, res2)[[1,3],:], 8, order = 'F')]), axis = 0)
    return(res)

def plotCI(res,  x, xscale = "linear", basex = 2):
    plt.figure(figsize=(10,10))
    plotLabel = ["val_acc", "train_acc", "num of iteration", "running time"]
    for i in range(4):
        plt.subplot(2, 2, (i + 1))
        plt.errorbar(x, res[:,(6-2*i)], yerr=res[:,(7-2*i)], fmt='.k');
        plt.title(plotLabel[i])
        if(xscale == "linear"): 
            plt.xscale("linear")
        elif(xscale == "log"): 
            plt.xscale('log', basex=basex)
            
def TablePerf(res, row, col, val = 0):
    if(val == 0): attr = "Accuracy of validation set"
    elif(val == 1): attr = "Accuracy of training set"
    elif(val == 2): attr = "Number of iteration"
    elif(val == 3): attr = "Running time"
    elif(val == 4):
        mean1 = np.reshape(res, (len(row), len(col), 8))[:,:,6-2*0].reshape(len(row), len(col))
        df = pd.DataFrame(mean1, columns = col, index = row).round(4)
        return df.stack().index[np.argmax(df.values)]
    mean = np.reshape(res, (len(row), len(col), 8))[:,:,6-2*val].reshape(len(row), len(col))
    size = np.reshape(res, (len(row), len(col), 8))[:,:,7-2*val].reshape(len(row), len(col))
    ui = mean + size
    li = mean - size

    df = pd.DataFrame(mean, columns=col, index=row).round(4)
    dfm = df.applymap(str)
    dfu = pd.DataFrame(ui, columns=col, index=row).round(4).applymap(str)
    dfl = pd.DataFrame(li, columns=col, index=row).round(4).applymap(str)
    
    dfm = dfm.apply(lambda x: x + " (" + dfl[x.name] + ", "+ dfu[x.name] + ")" )
    if(val ==  0 or val == 1): 
        bool_matrix = df == df.max().max()
        tmp = df.stack().index[np.argmax(df.values)]
    elif(val == 2 or val == 3):
        bool_matrix = df == df.min().min()
        tmp = df.stack().index[np.argmin(df.values)]
    def highlight(value):
        return bool_matrix.applymap(lambda x: 'background-color: yellow' if x else '')
    
    #print("The best " + attr + " is attained when # of hidden layers = %g , # of hidden units = %g\n" % tmp)

    print(attr)
    return(tmp, dfm.style.apply(highlight, axis=None))

def hyperpara(M, loss, activation, scale, nHiddenlayers, nHiddenunits,
             lr, momentum, batch_size):
    attr = ["accuracy of validation set", "accuracy of training set", "number of iteration", "running time"]
    
    res = performance(M, loss = loss, activation = activation, scale = scale)
    plotCI(res, scale, xscale = "log", basex = scale[1] / scale[0])
    plt.show()
    scale1 = scale[np.argmax(res[:,6])]
    print("The best scale hyper parameter is scale = %g" % scale1)

    res = performance(M, loss = loss, activation = activation, 
                      nHiddenlayers = nHiddenlayers, nHiddenunits = nHiddenunits, 
                      scale = [scale1])
    for i in range(4):
        a , b = TablePerf(res, val = i, row = nHiddenlayers, col = nHiddenunits); display(b)
        print("The best " + attr[i] + " is attained when # of hidden layers = %g , # of hidden units = %g\n" % a)
    nHiddenlayers1, nHiddenunits1 = TablePerf(res, val = 4, row = nHiddenlayers, col = nHiddenunits);
    print("The best number of hidden layers is nHiddenlayers = %g" % nHiddenlayers1)
    print("The best number of hidden units is nHiddenunits = %g" % nHiddenunits1)

    res = performance(M, loss = loss, activation = activation, scale = [scale1], 
                  nHiddenlayers = [nHiddenlayers1], nHiddenunits = [nHiddenunits1],
                  lr = lr, momentum = momentum)
    for i in range(4):
        a , b = TablePerf(res, val = i, row = lr, col = momentum); display(b)
    lr1, momentum1 = TablePerf(res, val = 4, row = lr, col = momentum);
    print("The best number of learning rate is eta = %g" % lr1)
    print("The best number of momentum is alpha = %g" % momentum1)

    res = performance(M, loss = loss, activation = activation, scale = [scale1], 
                  nHiddenlayers = [nHiddenlayers1], nHiddenunits = [nHiddenunits1],
                  lr = [lr1], momentum = [momentum1], batch_size = batch_size)

    plotCI(res, batch_size, xscale = "log", basex = batch_size[1] / batch_size[0])
    plt.show()
    batch_size1 = batch_size[np.argmax(res[:,6])]
    print("The best batch size hyper parameter is batch_size = %g" % batch_size1)

    return scale1, nHiddenlayers1, nHiddenunits1, lr1, momentum1, batch_size1

def perform(loss, activation, scale, nHiddenlayers, nHiddenunits, lr ,
                momentum, batch_size, verbose = 0):
    model = keras.Sequential()

    for i in range(int(nHiddenlayers)):
        model.add(keras.layers.Dense(nHiddenunits, activation=activation))

    model.add(keras.layers.Dense(10, activation='softmax'))
    
    model.compile(optimizer=keras.optimizers.SGD(lr = lr, momentum = momentum),
              loss= loss,
              metrics=['accuracy'])

    callbacks = [
        # Interrupt training if `val_loss` stops improving for over 2 epochs
        tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss')
        ]
    
    modelFit = model.fit(train_images*scale, encoded_train_labels , epochs=100, callbacks=callbacks,
         validation_data=(val_images*scale, encoded_val_labels), verbose = verbose, batch_size = batch_size)
    
    predictions=model.predict_classes(test_images)
    
    overallacc = accuracy_score(test_labels, predictions)
    classacc = classification_report(test_labels, predictions)
    con_mat = tf.math.confusion_matrix(labels=test_labels, predictions=predictions).numpy()
    con_mat = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)
    
    predictions2 = model.predict_classes(train_images_full)
    
    overallacc2 = accuracy_score(train_labels_full, predictions2)
    classacc2 = classification_report(train_labels_full, predictions2)
    con_mat2 = tf.math.confusion_matrix(labels=train_labels_full, predictions=predictions2).numpy()
    con_mat2 = np.around(con_mat2.astype('float') / con_mat2.sum(axis=1)[:, np.newaxis], decimals=2)
    
    return overallacc, classacc, con_mat, overallacc2, classacc2, con_mat2, model.summary()

def validation2(filtersize, height = 2, nconv = 1, pool_size = 2,
                lr = 0.01, epoch = 5,
                momentum = 0.0, batch_size = 32, verbose = 0):
    #loss = "categorical_crossentropy"
    
    model = keras.Sequential()
    
    model.add(keras.layers.Conv2D(filtersize, (height, height), activation='relu', input_shape=(8, 8, 1)))
    model.add(keras.layers.MaxPooling2D((pool_size, pool_size)))

    if(nconv == 2):
        model.add(keras.layers.Conv2D(filtersize, (height, height), activation='relu'))
        model.add(keras.layers.MaxPooling2D((pool_size, pool_size)))

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(10, activation='softmax'))
    
    model.compile(optimizer=keras.optimizers.SGD(lr = lr, momentum = momentum),
              loss="categorical_crossentropy",
              metrics=['accuracy'])

    callbacks = [
        # Interrupt training if `val_loss` stops improving for over 2 epochs
        tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss')
        ]
    
    start = timer()
    modelFit = model.fit(train_cnn, encoded_train_labels_cnn , epochs=epoch, callbacks=callbacks,
    validation_data=(val_cnn, encoded_val_labels_cnn), verbose = verbose, batch_size = batch_size)
    end = timer()
    
    time = end - start
    num_iter = len(modelFit.history['loss'])
    
    train_acc = model.evaluate(train_cnn, encoded_train_labels_cnn, verbose=0)[1]
    val_acc = model.evaluate(val_cnn, encoded_val_labels_cnn, verbose=0)[1]
    
    return time, num_iter, train_acc, val_acc

def perform2(filtersize, height, nconv, pool_size,
                lr, epoch,
                momentum, batch_size, verbose = 0):
    model = keras.Sequential()
    
    model.add(keras.layers.Conv2D(filtersize, (height, height), activation='relu', input_shape=(8, 8, 1)))
    model.add(keras.layers.MaxPooling2D((pool_size, pool_size)))

    if(nconv == 2):
        model.add(keras.layers.Conv2D(filtersize, (height, height), activation='relu'))
        model.add(keras.layers.MaxPooling2D((pool_size, pool_size)))

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(10, activation='softmax'))
    
    model.compile(optimizer=keras.optimizers.SGD(lr = lr, momentum = momentum),
              loss="categorical_crossentropy",
              metrics=['accuracy'])

    callbacks = [
        # Interrupt training if `val_loss` stops improving for over 2 epochs
        tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss')
        ]
    
    modelFit = model.fit(train_cnn, encoded_train_labels_cnn, epochs=epoch, callbacks=callbacks,
    validation_data=(val_cnn, encoded_val_labels_cnn), verbose = verbose, batch_size = batch_size)
    
    predictions=model.predict_classes(test_cnn)
    
    overallacc = accuracy_score(test_labels, predictions)
    classacc = classification_report(test_labels, predictions)
    con_mat = tf.math.confusion_matrix(labels=test_labels, predictions=predictions).numpy()
    con_mat = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)
    
    predictions2 = model.predict_classes(train_cnn_full)
    
    overallacc2 = accuracy_score(train_labels_full, predictions2)
    classacc2 = classification_report(train_labels_full, predictions2)
    con_mat2 = tf.math.confusion_matrix(labels=train_labels_full, predictions=predictions2).numpy()
    con_mat2 = np.around(con_mat2.astype('float') / con_mat2.sum(axis=1)[:, np.newaxis], decimals=2)
    
    return overallacc, classacc, con_mat, overallacc2, classacc2, con_mat2, model.summary()

def printpara1(scale1, nHiddenlayers1, nHiddenunits1, lr1, momentum1, batch_size1):
    print("scale = %g" % scale1)
    print("nHiddenlayers = %g, nHiddenunits = %g" % (nHiddenlayers1, nHiddenunits1))
    print("lr = %g, momentum = %g" % (lr1, momentum1))
    print("batch_size = %g" % batch_size1)
    return 0

def printpara2(filtersize5, nconv5, height5, pool_size5, batch_size5, lr5, momentum5):
    print("filtersize = %g, number of convolution layers = %g" % (filtersize5, nconv5))
    print("height of kernel= %g, pool_size = %g" % (height5, pool_size5))
    print("lr = %g, momentum = %g" % (lr5, momentum5))
    print("batch_size = %g" % batch_size5)
    return 0

M = 5
scale = [1/100, 1/10, 1, 10, 100]
nHiddenlayers = [0, 1, 2]
nHiddenunits = [1, 5, 10, 50, 100, 500, 1000]
lr = [0.001, 0.01, 0.1, 1]
momentum = [0, 0.001, 0.01, 0.1, 1]
batch_size = [8, 16, 32, 64, 128]

print("------------------Fully-connected feed-forward network------------------------")
print("------------------Sun-of-Squares; Relu------------------------")

#scale1, nHiddenlayers1, nHiddenunits1, lr1, momentum1, batch_size1 = hyperpara(
#    M = M, loss = "mse", activation = "relu", 
#    scale = scale, nHiddenlayers = nHiddenlayers, nHiddenunits = nHiddenunits, 
#    lr = lr, momentum = momentum, batch_size = batch_size
#)

scale1, nHiddenlayers1, nHiddenunits1, lr1, momentum1, batch_size1 = [1, 2, 1000, 0.1, 0.0, 8]
printpara1(scale1, nHiddenlayers1, nHiddenunits1, lr1, momentum1, batch_size1)
overallacc_test1, classacc_test1, con_mat_test1, overallacc_train1, classacc_train1, con_mat_train1, model_summary = perform(
    loss = "mse", activation = "relu", scale = scale1, 
    nHiddenlayers = nHiddenlayers1, nHiddenunits = nHiddenunits1, 
    lr = lr1, momentum = momentum1, batch_size = batch_size1)
print(model_summary)

print("Overall classification Accuracy for training data = %.4f\n" % overallacc_train1)
print("Class Accuracy for training data")
print(classacc_train1)
print("Confusion matrix for training data")
print(con_mat_train1)

print("-------------------------------------------------------------")

print("Overall classification Accuracy for test data = %.4f\n" % overallacc_test1)
print("Class Accuracy for test data")
print(classacc_test1)
print("Confusion matrix for test data")
print(con_mat_test1)

print("\n------------------Cross-entropy; Relu------------------------")

#scale2, nHiddenlayers2, nHiddenunits2, lr2, momentum2, batch_size2 = hyperpara(
#    M = M, loss = "categorical_crossentropy", activation = "relu", 
#    scale = scale, nHiddenlayers = nHiddenlayers, nHiddenunits = nHiddenunits, 
#    lr = lr, momentum = momentum, batch_size = batch_size
#)

scale2, nHiddenlayers2, nHiddenunits2, lr2, momentum2, batch_size2 = 1, 2, 1000, 0.01, 0.001, 8
printpara1(scale2, nHiddenlayers2, nHiddenunits2, lr2, momentum2, batch_size2)
overallacc_test2, classacc_test2, con_mat_test2, overallacc_train2, classacc_train2, con_mat_train2, model_summary = perform(
    loss = "categorical_crossentropy", activation = "relu", scale = scale2, 
    nHiddenlayers = nHiddenlayers2, nHiddenunits = nHiddenunits2, 
    lr = lr2, momentum = momentum2, batch_size = batch_size2)
print(model_summary)

print("Overall classification Accuracy for training data = %.4f\n" % overallacc_train2)
print("Class Accuracy for training data")
print(classacc_train2)
print("Confusion matrix for training data")
print(con_mat_train2)

print("-------------------------------------------------------------")

print("Overall classification Accuracy for test data = %.4f\n" % overallacc_test2)
print("Class Accuracy for test data")
print(classacc_test2)
print("Confusion matrix for test data")
print(con_mat_test2)

print("------------------tanh; Relu------------------------")

#scale3, nHiddenlayers3, nHiddenunits3, lr3, momentum3, batch_size3 = hyperpara(
#    M = M, loss = "categorical_crossentropy", activation = "tanh", 
#    scale = scale, nHiddenlayers = nHiddenlayers, nHiddenunits = nHiddenunits, 
#    lr = lr, momentum = momentum, batch_size = batch_size
#)

scale3, nHiddenlayers3, nHiddenunits3, lr3, momentum3, batch_size3 = [1, 2, 1000, 0.01, 0.0, 32]
printpara1(scale3, nHiddenlayers3, nHiddenunits3, lr3, momentum3, batch_size3)

overallacc_test3, classacc_test3, con_mat_test3, overallacc_train3, classacc_train3, con_mat_train3, model_summary = perform(
    loss = "categorical_crossentropy", activation = "tanh", scale = scale3, 
    nHiddenlayers = nHiddenlayers3, nHiddenunits = nHiddenunits3, 
    lr = lr3, momentum = momentum3, batch_size = batch_size3)
print(model_summary)

print("Overall classification Accuracy for training data = %.4f\n" % overallacc_train3)
print("Class Accuracy for training data")
print(classacc_train3)
print("Confusion matrix for training data")
print(con_mat_train3)

print("-------------------------------------------------------------")

print("Overall classification Accuracy for test data = %.4f\n" % overallacc_test3)
print("Class Accuracy for test data")
print(classacc_test3)
print("Confusion matrix for test data")
print(con_mat_test3)

print("\n------------------Convolution Neural Network------------------------")

#filtersize = [5, 10, 20]
#nconv = [1,1,1,1,1,1,1,1,2,2,2]
#height = [1,1,2,2,2,3,3,3,1,2,2]
#pool_size = [2,3,1,2,3,1,2,3,2,1,2]
#
#res = np.empty((0, 8))
#for i in range(len(filtersize)):
#    for j in range(len(nconv)):
#        #print(i, j)
#        tmp1, tmp2, tmp3, tmp4 = validation2(filtersize = filtersize[i], nconv = nconv[j],
#                                           height = height[j], pool_size = pool_size[j])
#        res = np.append(res, np.array([(filtersize[i], nconv[j], height[j], pool_size[j], #
#                                       tmp1, tmp2, tmp3, tmp4)]), axis = 0)
#        
#res2 = pd.DataFrame(res, columns=["filtersize", "nconv", "height", "pool_size",
#                                "running time", "numofIter", "train_acc", "val_acc"]).round(4)
#print(res2.to_string(index=False))
#
#print(res2.iloc[[res[:,7].argmax()]].to_string(index=False))
#filtersize5, nconv5, height5, pool_size5 = res[res[:,7].argmax(),:4].reshape(4).astype("int")
#
#batch_size = [16, 32]
#lr = [0.01, 0.1]
#momentum = [0, 0.01]
#
#res = np.empty((0, 7))
#for i in range(len(batch_size)):
#    for j in range(len(lr)):
#        for k in range(len(momentum)):
#            tmp1, tmp2, tmp3, tmp4 = validation2(filtersize = filtersize5, nconv = nconv5,
#                                             height = height5, pool_size = pool_size5,
#                                            batch_size = batch_size[i], lr = lr[j], momentum = momentum[k])
#            res = np.append(res, np.array([(batch_size[i], lr[j], momentum[k],
#                                       tmp1, tmp2, tmp3, tmp4)]), axis = 0)
#            
#res2 = pd.DataFrame(res, columns=["batch_size", "lr", "momentum",
#                                "running time", "numofIter", "train_acc", "val_acc"]).round(4)
#print(res2.to_string(index=False))
#
#print(res2.iloc[[res[:,6].argmax()]].to_string(index=False))
#batch_size5, lr5, momentum5 = res[res[:,6].argmax(),:3].reshape(3)
#batch_size5 = int(batch_size5)

filtersize5, nconv5, height5, pool_size5 = [20, 2, 2, 1]
batch_size5, lr5, momentum5 = [32, 0.1, 0.0]

printpara2(filtersize5, nconv5, height5, pool_size5, batch_size5, lr5, momentum5)

overallacc_test5, classacc_test5, con_mat_test5, overallacc_train5, classacc_train5, con_mat_train5, model_summary = perform2(
    filtersize = filtersize5, height = height5, nconv = nconv5, pool_size = pool_size5,
                lr = lr5, epoch = 100,
                momentum = momentum5, batch_size = batch_size5
)
print(model_summary)

print("Overall classification Accuracy for training data = %.4f\n" % overallacc_train5)
print("Class Accuracy for training data")
print(classacc_train5)
print("Confusion matrix for training data")
print(con_mat_train5)

print("-------------------------------------------------------------")

print("Overall classification Accuracy for test data = %.4f\n" % overallacc_test5)
print("Class Accuracy for test data")
print(classacc_test5)
print("Confusion matrix for test data")
print(con_mat_test5)



