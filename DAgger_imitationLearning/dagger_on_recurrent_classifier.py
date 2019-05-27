#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 02:11:48 2019

@author: abhijay
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
import seaborn as sns
sns.set()

class recurrent_classifier:
    
    def __init__(self, train_path):        
        self.seq_x, self.seq_y, self.labels, self.noOfLabels = self.get_data(train_path)
        self.test_seq_x, self.test_seq_y, self.test_labels, self.test_noOfLabels = self.get_data(test_path)
    
    # For generating features from the data
    def make_feature_y(self, t, x, y, is_yhat=False):
        
        if t-(self.historyLen)<0:
            y_history = y[0:t]
        else:
            y_history = y[t-(self.historyLen):t]
    
        y_history = np.pad(y_history, pad_width=(0,self.historyLen-len(y_history)), mode='constant', constant_values=(0))
        
        y_history_ = np.zeros((len(y_history),len(self.labels)))
        
        for i, ind in enumerate(y_history):
            if ind>0:
                y_history_[i,ind-1] = 1
        
        f_n = np.append( x[t], y_history_)
        
        if is_yhat:
            return f_n
        else:
            return ( f_n, y[t])
    
    # For making the instances
    def process_data(self, seq_x, seq_y):
        
        L = [] # Initialize set of classification examples
        
        for i, (x, y) in enumerate(zip( seq_x, seq_y)):            
        
            L_i = []            
            for t in range(x.shape[0]):
                L_i_ = self.make_feature_y( t, x, y)
                L_i.append( L_i_)
                            
            L.append(L_i)
        
        return L
    
    # For converting the instances to a feature set X
    def make_XY( self, L):
        X = []
        Y = []
        for seq in L:
            for (x,y) in seq:
                X.append(x)
                Y.append(y)
        X = np.array(X)
        Y = np.array(Y)
        return X, Y
    
    # For training on a linear classifier
    def learn_classifier( self, L):
        
        # Convert the instances to a feature set
        X, Y = self.make_XY( L)
        
        linearClassifier = svm.SVC(kernel='linear', gamma='scale')
        
        print ("\nTraining on.....", X.shape)
        fit  = linearClassifier.fit(X, Y)
        
        return linearClassifier
    
    # For calculating the recurrent error
    def calc_recurrent_error(self, L, linearClassifier):
        
        error = []
        mistakes = []
        for seq in L:
            X=[]
            Y=[]
            for (x,y) in seq:
                X.append(x)
                Y.append(y)
            
            # Predict on each example in the sequence
            Y_hat = linearClassifier.predict(np.array(X))
        
            error.append(sum(Y_hat-Y!=0)/len(Y))
            mistakes.append(sum(Y_hat-Y!=0))
        
        return np.mean( error), sum(mistakes)
    
    def calc_iid_error(self, L, linearClassifier):
        X, Y = self.make_XY(L)
        Y_hat = linearClassifier.predict(X)
        error = sum(Y_hat-Y!=0)/len(Y)
        # print ("IID Error: ", error)
        return error
    
    def exact_imitation( self):
        
        # For training data        
        L = self.process_data( self.train_seq_x, self.train_seq_y)
        
        # h = Classifier Learner
        linearClassifier = self.learn_classifier(L)
        
        return linearClassifier, L
    
    def learning_via_dagger( self, L_ei, beta_j):
        
        d_max = 5
        val_error = []
        val_mistakes = []
        
        test_error = []
        test_mistakes = []
        
        print ("\n===== beta_j ===== ",beta_j)
        
        print ("===== Training on L_ei =====")
        # h = Classifier Learner
        H_hat = self.learn_classifier(L_ei)

        # Best H_hat
        best_h_hat = H_hat
        val_error_lowest = 100
        L_best = L_ei.copy()
        
        print ("\n===== Applying Dagger =====")
        L = L_ei.copy()
        for dagger_iteration in range(d_max):
            
            # For training data        
            for i, (x, y) in enumerate(zip( self.train_seq_x, self.train_seq_y)):
                
                y_hat = []
                # For each example in the sequence
                for t in range(x.shape[0]):
                    L_i = self.make_feature_y( t, x, y_hat, True)
                    
                    # The policy we are following
                    if np.random.random_sample() >= beta_j:
                        y_hat.append(H_hat.predict(np.array([L_i]))[0])
                    else:
                        y_hat.append(y[t])
                    
                    # Aggregate data
                    if y_hat[t] != y[t]:
                        L.append([( L_i, y[t])])
            
            # Train a classifier after data aggregation
            H_hat = self.learn_classifier(L)
            
            val_error_, val_mistakes_ = self.calc_recurrent_error( self.L_val, H_hat)
            val_error.append(val_error_)
            val_mistakes.append(val_mistakes_)
            print ("Recurrent Val Error: ", val_error_)

            if val_error_ < val_error_lowest:
            	val_error_lowest = val_error_
            	best_h_hat = H_hat
            	L_best = L.copy()

            test_error_, test_mistakes_ = self.calc_recurrent_error( self.L_test, H_hat)
            test_error.append(test_error_)
            test_mistakes.append(test_mistakes_)            
            print ("Recurrent Test Error: ", test_error_)
            
            # decay
            # beta_j *= 0.85
        
        return best_h_hat, L_best, val_error, val_mistakes, test_error, test_mistakes
    
    # For getting the data
    def get_data(self, path):
        
        with open(path, "r") as f:
            lines = f.readlines()

        seq_x = []
        seq_y = []
        seq_x_ = []
        seq_y_ = []
        labels = set()
        
        for line in lines:
            line = line.strip().split("\t")
            if line[0] == '':
                if seq_x_:
                    labels.update(seq_y_)
                    seq_x.append(np.array(seq_x_))
                    seq_y.append(np.array(seq_y_))
                seq_x_ = []
                seq_y_ = []
                continue
            else:
                seq_x_.append( np.array([int(i) for i in line[1][2:]]))
                seq_y_.append( line[2])

        labels = list(labels)
        labels.sort()
        noOfLabels = len(labels)
        
        return seq_x, seq_y, labels, noOfLabels
    
    def plot( self, ei_value, y_values, yLabel, name, saveAs):
        plt.figure()
        ax = sns.lineplot(x=np.arange( 1, 6), y=np.repeat( ei_value, 5), label="Exact Imitation", dashes=True)
        for i in np.arange( 0, 5):
            ax = sns.lineplot(x=np.arange( 1, 6), y=y_values[i], label="Dagger, beta = "+str( round((i+5)*0.1,1)), markers=True)
        ax.set_title(name)
        ax.set_xlabel("Iterations")
        ax.set_ylabel(yLabel)
        box = ax.get_position()
        ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        ax.set_position([box.x0, box.y0, box.width * 0.65, box.height])
        # ax.set_ylim(min(min(y_values))-np.mean(y_values)/10,max(max(y_values))+np.mean(y_values)/10)
        ax.figure.savefig("plots/"+saveAs+".png")
     
if __name__ == "__main__":
    
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset', help='Specifies the dataset to use')
    args=parser.parse_args()

    dataset = args.dataset
        
    # dataset = 'nettalk_stress'
    # dataset = 'ocr_fold0_sm'

    train_path = "datasets/"+dataset+"_train.txt"
    test_path = "datasets/"+dataset+"_test.txt"

    dataset = dataset.split('_')[0]

    print ("Dataset: "+dataset)

    # Get data
    recurrentClassifier = recurrent_classifier( train_path)
    recurrentClassifier.dataset = dataset    

    # Preprocessing
    l = []
    for seq in recurrentClassifier.seq_x:
        l.append(len(seq))

    seq_y=[]
    for y in recurrentClassifier.seq_y:
        seq_y.append([recurrentClassifier.labels.index(y_)+1 for y_ in y])

    recurrentClassifier.seq_y = seq_y

    seq_y=[]
    for y in recurrentClassifier.test_seq_y:
        seq_y.append([recurrentClassifier.labels.index(y_)+1 for y_ in y])

    recurrentClassifier.test_seq_y = seq_y

    # Decide y_history length in the feature generation
    recurrentClassifier.historyLen = 2

    # Separate a validation set from the training dataset
    if dataset == 'nettalk':
        trainLen = int(0.9*len(recurrentClassifier.seq_x))
        recurrentClassifier.train_seq_x = recurrentClassifier.seq_x[0:trainLen]
        recurrentClassifier.train_seq_y = recurrentClassifier.seq_y[0:trainLen]        
        recurrentClassifier.val_seq_x = recurrentClassifier.seq_x[trainLen+1::]
        recurrentClassifier.val_seq_y = recurrentClassifier.seq_y[trainLen+1::]
        
    elif dataset == 'ocr':
        trainLen = int(0.1*len(recurrentClassifier.seq_x))
        recurrentClassifier.train_seq_x = recurrentClassifier.seq_x[trainLen+1::]
        recurrentClassifier.train_seq_y = recurrentClassifier.seq_y[trainLen+1::]        
        recurrentClassifier.val_seq_x = recurrentClassifier.seq_x[0:trainLen]
        recurrentClassifier.val_seq_y = recurrentClassifier.seq_y[0:trainLen]
        
    print ("===== Perform exact_imiattion =====")
    learned_classifier, L = recurrentClassifier.exact_imitation()

    # For validation data
    recurrentClassifier.L_val = recurrentClassifier.process_data( recurrentClassifier.val_seq_x, recurrentClassifier.val_seq_y)
    val_error_ei, val_mistakes_ei = recurrentClassifier.calc_recurrent_error( recurrentClassifier.L_val, learned_classifier)
    print ("\nRecurrent Error on Val data:\n", val_error_ei)
    print ("\nIID Val Error on Val data:\n", recurrentClassifier.calc_iid_error( recurrentClassifier.L_val, learned_classifier))

    # For test data
    recurrentClassifier.L_test = recurrentClassifier.process_data( recurrentClassifier.test_seq_x, recurrentClassifier.test_seq_y)
    test_error_ei, test_mistakes_ei = recurrentClassifier.calc_recurrent_error( recurrentClassifier.L_test, learned_classifier)
    print ("\nRecurrent Test Error:\n", test_error_ei)
    print ("\nIID Test Error:\n", recurrentClassifier.calc_iid_error( recurrentClassifier.L_test, learned_classifier))

    print ("\n===== learning_via_dagger =====")
    val_error = []
    val_mistakes =[]
    test_error = []
    test_mistakes = []
    for beta in np.linspace(0.5, 0.9, 5):
        classifier, L_i, val_error_, val_mistakes_, test_error_, test_mistakes_ = recurrentClassifier.learning_via_dagger( L.copy(), beta)
        val_error.append(val_error_)
        val_mistakes.append(val_mistakes_)
        test_error.append(test_error_)
        test_mistakes.append(test_mistakes_)

    recurrentClassifier.plot(val_error_ei, val_error, "Error rate", "Recurrent Error on Val_data ("+dataset+")", dataset+"/val_data_recurrent_error")
    recurrentClassifier.plot(test_error_ei, test_error, "Error rate", "Recurrent Error on Test_data ("+dataset+")", dataset+"/test_data_recurrent_error")
    recurrentClassifier.plot(val_mistakes_ei, val_mistakes, "Mistakes", "Mistakes on Val_data ("+dataset+")", dataset+"/val_data_mistakes")
    recurrentClassifier.plot(test_mistakes_ei, test_mistakes, "Mistakes", "Mistakes on Test_data ("+dataset+")", dataset+"/test_data_mistakes")