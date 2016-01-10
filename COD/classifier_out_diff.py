#! /usr/bin/env python

import shutil
import tempfile
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.datasets import fetch_mldata

def computeCOD(nb_pred, svm_pred, targets, N):
    # number of instances where h1 is correct, but h2 is incorrect
    n10 = 0
    # number of instances where h2 is correct, but h1 is incorrect
    n01 = 0
    # number of instances where both h1 and h2 are uncorrect, and they make different predictions
    n00D = 0

    for pred1, pred2, t in zip(nb_pred, svm_pred, targets):
        n10 += 1 if pred1 == t and pred2 != t else 0
        n01 += 1 if pred1 != t and pred2 == t else 0
        n00D += 1 if pred1 != t and pred2 != t and pred1 != pred2 else 0

    return float(n10 + n01 + n00D) / float(N)

''' This version of the program loads multiple datasets
    from the intertet using fetch_mldata method '''
if __name__ == "__main__":

    # Define learning algorithms
    gauss_nb = GaussianNB()
    s_vector_machine = SVC()
    custom_data_home = tempfile.mkdtemp()

    cod_vector = list()
    datasets_file = open('/home/salt/Desktop/cs676/datasets.txt')

    for dataset_name in datasets_file:
        # Download the necessary dataset
        print dataset_name.strip()
        dataset = fetch_mldata(dataset_name.strip(), data_home=custom_data_home)
        try: 
            # Naive Bayes
            nb_pred = gauss_nb.fit(dataset.data, dataset.target).predict(dataset.data)
            # Support Vector Machine 
            svm_pred = s_vector_machine.fit(dataset.data, dataset.target).predict(dataset.data)
        except: 
            print ("Something went wrong with %s dataset"
                % (dataset_name.strip().upper()))
            continue

        COD = computeCOD(nb_pred, svm_pred, dataset.target, dataset.data.shape[0])
        cod_vector.append(COD)
        print ("Classifier output difference for dataset %s is %f"
            % (dataset_name.upper(), COD))
    
    print ("Average COD between Naive Bayes and Support Vector Machine: %f"
            % (sum(cod_vector) / float(len(cod_vector))))
    
    # Clean up the temporary directory
    shutil.rmtree(custom_data_home)
    datasets_file.close()
