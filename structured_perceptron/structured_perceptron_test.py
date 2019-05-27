#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 02:11:48 2019

@author: abhijay
"""

import argparse
import numpy as np
import random
from nltk.util import ngrams
import pickle
import matplotlib.pyplot as plt
# import time

class structured_preceptron:
    
    def __init__(self, test_path):
        
        self.seq_x, self.seq_y = self.get_data(test_path)

    def phi_( self, x, y):
        
        # unary features
        unary_features = np.zeros(( self.noOfLabels, x.shape[1]))
        
        for x_, y_ in zip( x, y):
            unary_features[self.labels.index(y_),:] += x_
            
        features = unary_features.flatten()
        
        if self.order > 0:
            # pairwise features
            pairwise_features = np.zeros(( self.noOfLabels, self.noOfLabels))
            for pair in list(ngrams( y, 2)):
                pairwise_features[ self.labels.index(pair[0]), self.labels.index(pair[1])] += 1
            
            features = np.concatenate([features, pairwise_features.flatten()])
            
            if self.order > 1:
                # triple features
                triple_features = np.zeros(( self.noOfLabels, self.noOfLabels, self.noOfLabels))
                for pair in list(ngrams( y, 3)):
                    triple_features[ self.labels.index(pair[0]), self.labels.index(pair[1]), self.labels.index(pair[2])] += 1    
                
                features = np.concatenate([ features, triple_features.flatten()])
                
                if self.order > 2:
                    # quadruple features
                    quadruple_features = np.zeros(( self.noOfLabels, self.noOfLabels, self.noOfLabels, self.noOfLabels))
                    for pair in list(ngrams( y, 4)):
                        quadruple_features[ self.labels.index(pair[0]), self.labels.index(pair[1]), self.labels.index(pair[2]), self.labels.index(pair[3])] += 1
                    
                    features = np.concatenate([ features, quadruple_features.flatten()])
        
        return features
    
    def rgs_inference( self, x, w, R):
        
        # random initialization
        y_hat_rand = [self.labels[j] for j in [round(random.uniform(0,len(self.labels)-1)) for i in range(x.shape[0])]]
        y_hat = y_hat_rand.copy()
        
        features_phi = self.phi_( x, y_hat_rand)
        
        S_best = np.dot( w, features_phi)
        
        # Variant while testing, not a random start IMP TRY TRY TRY
        for r in range(R):
            while True:
                prev_S_best = S_best
                
                # change one label
                for index in range(len(y_hat_rand)):
                    for change in self.labels:
                        y_hat_changed = y_hat_rand.copy()
                        y_hat_changed[index] = change
                        
                        features_phi = self.phi_( x, y_hat_changed)

                        S = score( w, features_phi)
                        
                        if S > S_best:
                            S_best = S
                            y_hat = y_hat_changed.copy()
    
                y_hat_rand = y_hat.copy()
                
                # No change in predicted sequence from previous best
                # Reached local minima
                if prev_S_best == S_best: break
            
            # Take a new random start
            y_hat_rand = [self.labels[j] for j in [round(random.uniform(0,len(self.labels)-1)) for i in range(x.shape[0])]]
            
        return y_hat
    
    def onlineStructuredPerceptron_driver(self):
        
        test_accuracies = []
        for iteration in range(self.MAX_ITERATIONS):
            
            w = load_weights("saved/"+self.dataset+"/"+self.question+"/weights_"+str(self.experiment)+"_"+str(iteration+1)+".pkl")
            
            accuracies = []
            for i, (x, y) in enumerate(zip( self.seq_x, self.seq_y)):
                
                y_hat = self.rgs_inference( x, w, self.R)
                accuracies.append(sum(y_hat == y)/len(y))
            
            mean_accuracy = np.mean(accuracies) # (1/n)*sum(hamming_accuracies)
            print ("Iteration: "+str(iteration+1)+", Accuracy: "+str(mean_accuracy))
            test_accuracies.append(mean_accuracy)
                        
        return test_accuracies
    
    def get_data(self, path):
        
        with open(path, "r") as f:
            lines = f.readlines()

        seq_x = []
        seq_y = []
        seq_x_ = []
        seq_y_ = []
        
        for line in lines:
            line = line.strip().split("\t")
            if line[0] == '':
                if seq_x_:
                    seq_x.append(np.array(seq_x_))
                    seq_y.append(np.array(seq_y_))
                seq_x_ = []
                seq_y_ = []
                continue
            else:
                seq_x_.append( np.array([int(i) for i in line[1][2:]]))
                seq_y_.append( line[2])
        
        return seq_x, seq_y
    
    
    def plot_accuracies( self, test_accuracies, order, title, saveAs):
        
        fig, ax = plt.subplots(figsize=(10, 8))
        for i, accuracies in enumerate(test_accuracies):
            ax.plot( range(1,len(accuracies)+1), [accuracy*100 for accuracy in accuracies], label = order[i])
        ax.set_ylim( (np.min(test_accuracies)*100)-5, (np.max(test_accuracies)*100)+5)
        ax.set_title( title, fontsize=18)
        ax.set_ylabel( "Accuracy", fontsize=15)
        ax.set_xlabel( "Iteration", fontsize=15)
        ax.set_xticks( range(1,len(test_accuracies[0])+1))
        ax.legend(loc="lower right")
        fig.savefig("plots/"+saveAs)

def score( w, phi):
    return np.dot( w, phi)

def load_weights( weights_filename):
    with open(weights_filename, 'rb') as f:
        w = pickle.load(f)
    return w

     
if __name__ == "__main__":
    
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset', help='Specifies the dataset to use')
    args=parser.parse_args()
    
    dataset = args.dataset
    # dataset = 'nettalk_stress'
    test_path = "datasets/"+dataset+"_test.txt"
    
    dataset = dataset.split('_')[0]
    
    print ("Dataset: "+dataset)
    
    structuredPerceptron = structured_preceptron( test_path)
    structuredPerceptron.dataset = dataset
    
    with open( "saved/"+structuredPerceptron.dataset+"/"+"labels.txt", 'rb') as f:
        structuredPerceptron.labels = pickle.load( f)
    structuredPerceptron.noOfLabels = len(structuredPerceptron.labels)
    
    structuredPerceptron.MAX_ITERATIONS = 20
    structuredPerceptron.R = 5
    structuredPerceptron.ETA = 0.01
    
    structuredPerceptron.question = "q_1c"
    print ("Q_1c")
    test_accuracies = []
    orders = [ 'unary', 'unary+pairwise', 'unary+pairwise+triples', 'unary+pairwise+triples+quadruples']
    for i, order in enumerate(range(1,5)):
        print ("Experiment: "+str(orders[i]))
        test_accuracies.append([])
        structuredPerceptron.order = order
        structuredPerceptron.experiment = i
        accuracies = structuredPerceptron.onlineStructuredPerceptron_driver()
        test_accuracies[i] = accuracies
    
    structuredPerceptron.plot_accuracies( test_accuracies, orders, dataset+"_q1c_structured_perceptron_test_accuracies", dataset+"_q1c_structured_perceptron_test_accuracies.png")
    
    structuredPerceptron.question = "q_1d"
    print ("Q_1d")
    structuredPerceptron.order = 1
    test_accuracies = []
    restarts = [ 'R=1', 'R=5', 'R=10', 'R=20']
    for i, restart in enumerate([1, 5, 10, 20]):
        print ("Experiment: "+str(restarts[i]))
        test_accuracies.append([])
        structuredPerceptron.R = restart
        structuredPerceptron.experiment = i
        accuracies = structuredPerceptron.onlineStructuredPerceptron_driver()
        test_accuracies[i] = accuracies
    
    structuredPerceptron.plot_accuracies( test_accuracies, restarts, dataset+"_q1d_structured_perceptron_test_accuracies", dataset+"_q1d_structured_perceptron_test_accuracies.png")