#! /usr/bin/env python

#import shutil 
#import tempfile
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

def crossValidationSplit(seed, dataset, target_set, split=0.2):
    train_set, test_set, train_target, test_target = ([], [], [], [])

    rs = cross_validation.ShuffleSplit(len(dataset), n_iter=1, test_size=split, random_state=seed)


    for train_index, test_index in rs:
        for train_i, test_i in zip_long(train_index, test_index):
            # Get training set instances
            train_set.append(dataset[train_i])
            train_target.append(target_set[train_i])
            # Get test set instances
            if test_i != None:
               test_set.append(dataset[test_i])
               test_target.append(target_set[test_i])

    if split == 0.2:
        return train_set, test_set, train_target, test_target
    return train_set, train_target

def runClassifiers(train_set, train_target, test_data):
    # Decision Tree
    #gauss_nb = GaussianNB()
    #nb_pred = gauss_nb.fit(train_set, train_target).predict(test_data)

    dec_tree = DecisionTreeClassifier()
    dt_pred = dec_tree.fit(train_set, train_target).predict(test_data)

    return dt_pred


''' Load a general CSV file and return a Bunch data structure
    Assume last column is a target column '''
    
def loadCSV(dest_dir, file_name, i=0):

    with open(join(dest_dir, file_name+'.data')) as csv_file:
        #data_file = csv.reader(csv_file)
        raw_data = np.loadtxt(csv_file, delimiter=",").astype(float)

    n_samples = int(len(raw_data))
    n_features = int(len(raw_data)-1)

    data = raw_data[:, :-1]
    target = raw_data[:, -1]

    return Bunch(data=data, target=target)


def computeCOD(nb_pred1, nb_pred2, N):
    cod = 0
    for pred1, pred2 in zip(nb_pred1, nb_pred2):
        cod += 1 if pred1 != pred2 else 0

    return float(cod) / float(N)

#def computeCOD(nb_pred, dt_pred, targets, N):
#    # number of instances where h1 is correct, but h2 is incorrect
#    n10 = 0
#    # number of instances where h2 is correct, but h1 is incorrect
#    n01 = 0
#    # number of instances where both h1 and h2 are uncorrect, and they make different predictions
#    n00D = 0
#
#    for pred1, pred2, t in zip(nb_pred, dt_pred, targets):
#        n10 += 1 if pred1 == t and pred2 != t else 0
#        n01 += 1 if pred1 != t and pred2 == t else 0
#        n00D += 1 if pred1 != t and pred2 != t and pred1 != pred2 else 0
#
#    return float(n10 + n01 + n00D) / float(N)




if __name__ == "__main__":
    # Find out what methods the directory has
    # methods = [method for method in dir(datasets) if callable(getattr(datasets, method)) and "load" in method]
    #custom_data_home = tempfile.mkdtemp()
    dataset_dir_home = '/home/salt/Desktop/CS676/COD/datasets'
    
    dataset_names = ['sensorless_drive_diagnosis', 'diabetes', 'pima-indians-diabetes', 'iris']

    cod_vector = list()
    for name in dataset_names:
        # Get the dataset
        #dataset = datasets.fetch_mldata(name, data_home=custom_data_home)
        dataset = loadCSV(dataset_dir_home, name)
        print "Dataset {0} is loaded, shape {1}".format(name.upper().split('/')[-1], dataset.data.shape)

        dataset_cod_vector = list()
        # h1
        seed = randint(0, dataset.data.shape[0])
        train_set, test_set, train_target, test_target = crossValidationSplit(seed, dataset.data, dataset.target)

        nb_pred1 = runClassifiers(train_set, train_target, test_set)
        model_list = [nb_pred1]

        for i in xrange(10):
            # h2
            seed = randint(0, dataset.data.shape[0])
            train_set, train_target = crossValidationSplit(seed, train_set, train_target, 0)

            nb_pred2 = runClassifiers(train_set, train_target, test_set)
            
            # Compute pairwise COD
            t_cod_vector = list()
            for pred_model in model_list:
                COD = computeCOD(pred_model, nb_pred2, len(test_target))
                t_cod_vector.append(COD)
                cod_vector.append(COD)
                dataset_cod_vector.append(COD)

            model_list.append(nb_pred2)

            print "{0}".format(t_cod_vector)

        print ("Average COD for Decision Tree for dataset %s: %f"
                % (name.upper().split('/')[-1], sum(dataset_cod_vector) / float(len(dataset_cod_vector))))
        print # Separate datasets by new line
    
    print ("Average COD between Decision Tree: %f"
            % (sum(cod_vector) / float(len(cod_vector))))

    # Clean up the temporary directory
    #shutil.rmtree(custom_data_home)
