import ftn

# Input parameters
filtersize5, nconv5, height5, pool_size5 = [20, 2, 2, 1]
batch_size5, lr5, momentum5 = [32, 0.1, 0.0]

ftn.printpara2(filtersize5, nconv5, height5, pool_size5, batch_size5, lr5, momentum5)

overallacc_test5, classacc_test5, con_mat_test5, overallacc_train5, classacc_train5, con_mat_train5, model_summary = ftn.perform2(
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