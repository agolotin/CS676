#! /usr/bin/env python

import shutil
import tempfile
import csv

from sklearn import datasets
from sklearn.datasets.base import Bunch
from sklearn import cross_validation
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

from itertools import izip_longest as zip_long
from os.path import join

from random import randint
import numpy as np

def crossValidationSplit(seed, dataset):
    train_set, test_set, train_target, test_target = ([], [], [], [])

    rs = cross_validation.ShuffleSplit(dataset.data.shape[0], n_iter=1, test_size=0.2, random_state=seed)

    for train_index, test_index in rs:
        for train_i, test_i in zip_long(train_index, test_index):
            # Get training set instances
            train_set.append(dataset.data[train_i])
            train_target.append(dataset.target[train_i])
            # Get test set instances
            if test_i != None:
               test_set.append(dataset.data[test_i])
               test_target.append(dataset.target[test_i])

    #return (np.array(train_set).astype(float), np.array(test_set).astype(float), 
    #        np.array(train_target).astype(float), np.array(test_target).astype(float))
    return train_set, test_set, train_target, test_target

def runClassifiers(train_set, train_target, test_data, test_target):
    # Naive Bayes
    gauss_nb = GaussianNB()
    nb_model = gauss_nb.fit(train_set, train_target)

    nb_pred = nb_model.predict(test_data)
    print ("Naive Bayes accuracy: %f" %
            nb_model.score(test_data, test_target))

    # Decision tree
    dec_tree = DecisionTreeClassifier()
    dt_model = dec_tree.fit(train_set, train_target)

    dt_pred = dt_model.predict(test_data)
    print ("Decision Tree accuracy: %f" %
            dt_model.score(test_set, test_target))

    return nb_pred, dt_pred


''' Load a general CSV file and return a Bunch data structure
    Assume last column is a target column '''
    
def loadCSV(dest_dir, file_name, i=0):

    with open(join(dest_dir, file_name+'.data')) as csv_file:
        #data_file = csv.reader(csv_file)
        raw_data = np.loadtxt(csv_file, delimiter=",").astype(float)

    n_samples = int(len(raw_data))
    n_features = int(len(raw_data)-1)

#    data = raw_data[:, :-1]
#    target = raw_data[:, -1]

    data = np.empty((n_samples, n_features))
    target = np.empty((n_samples,))

    for sample in raw_data:
        data[i] = np.asarray(sample[:-1], dtype="float")
        target[i] = np.asarray(sample[-1], dtype="float")
        i += 1

    return Bunch(data=data, target=target)



def computeCOD(nb_pred, dt_pred, targets, N):
    # number of instances where h1 is correct, but h2 is incorrect
    n10 = 0
    # number of instances where h2 is correct, but h1 is incorrect
    n01 = 0
    # number of instances where both h1 and h2 are uncorrect, and they make different predictions
    n00D = 0

    for pred1, pred2, t in zip(nb_pred, dt_pred, targets):
        n10 += 1 if pred1 == t and pred2 != t else 0
        n01 += 1 if pred1 != t and pred2 == t else 0
        n00D += 1 if pred1 != t and pred2 != t and pred1 != pred2 else 0

    return float(n10 + n01 + n00D) / float(N)




if __name__ == "__main__":
    # Find out what methods the directory has
    # methods = [method for method in dir(datasets) if callable(getattr(datasets, method)) and "load" in method]
    custom_data_home = tempfile.mkdtemp()
    dataset_dir_home = '/home/salt/Desktop/cs676/COD/datasets'

    '''
    diabetes: (768 instances, 8 attributes)
    Twitter: (38393 instances, 77 attributes)
    HIGGS: (11000000 instances, 28 attributes)
    Gas sensor flow: (4178504 instances, 19 attribuets)
    Daily and Sports activity: (9120 instances, 5625 attributes)
    '''

    dataset_names = ['social_media/TomsHardware/TomsHardware','social_media/Twitter/Twitter', 'pima-indians-diabetes']#['MNIST Original', 'leukemia', 'diabetes']

    cod_vector = list()
    for name in dataset_names:
        # Get the dataset
        #dataset = datasets.fetch_mldata(name, data_home=custom_data_home)
        dataset = loadCSV(dataset_dir_home, name)
        print "Dataset %s is loaded" % name.upper().split('/')[-1]

        dataset_cod_vector = list()
        for i in xrange(10):

            seed = randint(0, dataset.data.shape[0])
            train_set, test_set, train_target, test_target = crossValidationSplit(seed, dataset)

            nb_pred, dt_pred = runClassifiers(train_set, train_target, test_set, test_target)

            COD = computeCOD(nb_pred, dt_pred, test_target, len(test_target))
            cod_vector.append(COD)
            dataset_cod_vector.append(COD)

            print ("Classifier Output Difference for dataset %s, seed %d is %f"
                    % (name.upper(), seed, COD))

        print ("Average COD between Naive Bayes and Decision Tree: %f"
                % (sum(dataset_cod_vector) / float(len(dataset_cod_vector))))
        print # Separate datasets by new line
    
    print ("Average COD between Naive Bayes and Decision Tree: %f"
            % (sum(cod_vector) / float(len(cod_vector))))

    # Clean up the temporary directory
    shutil.rmtree(custom_data_home)
