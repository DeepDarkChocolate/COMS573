'''
Created on Feb 5, 2020

@author: Yonghyun
'''
import numpy as np
import csv

if __name__ == '__main__':
    
    # 1. data import
    train_label = []
    with open("./20newsgroups/train_label.csv", "r") as f:
        rdr = csv.reader(f)
        for line in rdr:
            train_label.append(int(line[0]))
    len_train = len(train_label) # 11269
    
    test_label = []
    with open("./20newsgroups/test_label.csv", "r") as f:
        rdr = csv.reader(f)
        for line in rdr:
            test_label.append(int(line[0]))
    
    len_voca = 0
    with open("./20newsgroups/vocabulary.txt", "r") as f:
        rdr = csv.reader(f)
        for line in rdr:
            len_voca += 1 # 61188
    
    # 2.1 Learn the Naive Bayes Model
    prior = np.bincount(train_label)[1:] / len_train # p(omega_j)
    len_prior = len(prior) # 20
    
    # class prior P(omega_j)
    print("1. Class priors")
    for i in range(len_prior):
        print("P(Omega = %d)= %g" % (i + 1, prior[i]))
    print("")
    
    # n_k: number of times word w_k occurs in all documents in class omega_j
    nk = np.zeros(len_voca*len_prior).reshape(len_voca,len_prior)
    # n: total number of words in all documents in class omega_j
    n = np.zeros(len_prior)
    with open("./20newsgroups/train_data.csv", "r") as f:
        rdr = csv.reader(f)
        for line in rdr:
            docIdx = int(line[0]);
            wordIdx = int(line[1]);
            labelIdx = train_label[docIdx - 1]
            nk[wordIdx - 1, labelIdx - 1] += int(line[2])
            n[labelIdx - 1] += int(line[2])
    
    MLE = np.true_divide(nk, n)
    BE = np.true_divide(nk + 1, n + len_voca)
    MLE2 = np.where(MLE == 0, -1, 0)
    
    def prediction(stream, logE):
        res = []
        with stream as f:
            rdr = csv.reader(f)
            docIdx = 1
            vec = np.zeros(len_prior)
            for line in rdr:
                if(docIdx != int(line[0])):
                    res.append(np.argmax(np.log(prior) + vec) + 1)
                    docIdx += 1
                    vec = np.zeros(len_prior)
                vec = vec + int(line[2]) * logE[int(line[1]) - 1, ]
            res.append(np.argmax(np.log(prior) + vec) + 1)
        return res
    
    def OverallAccuracy(prediction, label):
        return np.sum(np.array(prediction) == np.array(label)) / len(label)
    
    def ClassAccuracy(prediction, label):
        return(np.bincount(np.array(label)[np.array(prediction) == np.array(label)], minlength = len_prior + 1)[1:] / np.bincount(label)[1:])
    
    def ConfusionMatrix(prediction, label):
        mat = np.zeros(len_prior*len_prior).reshape(len_prior,len_prior)
        for i in range(len_prior):
            mat[i:] = np.bincount(np.array(prediction)[np.array(label) == i + 1], minlength = len_prior + 1)[1:]
        return mat
        
    def PrintMatrix(mat):
        print('\n'.join([' '.join([str(cell) for cell in row]) for row in mat.astype(int)]))
    
    def PrintResult(prediction, label):
        print("Overall Accuracy = %g" % OverallAccuracy(prediction, label))
        print("Class Accuracy:")
        for i in range(len_prior):
            print("Group %d: %g" % (i + 1, ClassAccuracy(prediction, label)[i]))
        print("Confusion Matrix:")
        PrintMatrix(ConfusionMatrix(prediction, label))
        print("")
    
    print("2. Results based on Bayesian estimator")
    print("2.1. Training Data on Bayesian estimator")
    prediction11 = prediction(open("./20newsgroups/train_data.csv", "r"), np.log(BE))
    PrintResult(prediction11, train_label)
    
    print("2.2. Test Data on Bayesian estimator")
    prediction12 = prediction(open("./20newsgroups/test_data.csv", "r"), np.log(BE))
    PrintResult(prediction12, test_label)
    
    print("3. Results based on Maximum Likelihood estimator")
    print("3.1. Training Data of Maximum Likelihood estimator")
    prediction21 = prediction(open("./20newsgroups/train_data.csv", "r"), MLE2)
    PrintResult(prediction21, train_label)
    
    print("3.2. Test Data of Maximum Likelihood estimator")
    prediction22 = prediction(open("./20newsgroups/test_data.csv", "r"), MLE2)
    PrintResult(prediction22, test_label)

    
