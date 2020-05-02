import ftn

# Input parameters
loss = "mse"
activation = "relu"
scale1, nHiddenlayers1, nHiddenunits1, lr1, momentum1, batch_size1 = [1, 2, 1000, 0.1, 0.0, 8]

print("loss = %s" % loss)
print("activation = %s" % activation)
ftn.printpara1(scale1, nHiddenlayers1, nHiddenunits1, lr1, momentum1, batch_size1)
overallacc_test1, classacc_test1, con_mat_test1, overallacc_train1, classacc_train1, con_mat_train1, model_summary = ftn.perform(
    loss = loss, activation = activation, scale = scale1, 
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