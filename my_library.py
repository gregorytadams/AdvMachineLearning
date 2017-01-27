# my_library.py

from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import KFold

def get_model_evaluation_stats(y_true, y_pred, labels=None, pos_label=1, average='binary', \
                               sample_weight=None, title="Summary"):
    '''
    Custom summary function to get a quick look at my models.

    Requires import above: 'from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score'
    
    Inputs:
    (all) same inputs as sklearn.model_selection inputs of the same name.  Refer to sklearn documentation at 
    http://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics
    '''
    print("TITLE: {}".format(title))
    print("F1: {}".format(f1_score(y_true, y_pred, pos_label=pos_label, average=average, sample_weight=sample_weight)))
    print("Precision: {}".format(precision_score(y_true, y_pred, pos_label=pos_label, average=average, sample_weight=sample_weight)))
    print("Recall: {}".format(recall_score(y_true, y_pred, pos_label=pos_label, average=average, sample_weight=sample_weight)))
    print("Accuracy: {}".format(accuracy_score(y_true, y_pred, sample_weight=sample_weight)))
    print("-"*30)

def get_k_folds(data_list, target_list, k):
    '''
    Gets the k folds for cross validation.

    Inputs
    data_list, a list of lists, each sublist of which is a sentence
    target_list, a list of strings, each element of which is the label for data_list

    Outputs
    train_data and test_data:  a list of k folds, each of which is a list of lists with each sublist containing a sentence
    train_labels and test_labels: a list of lists, each sublist of which is comprised of strings that are labels
    '''
    kf = KFold(n_splits=k, shuffle=True, random_state=0)
    train_data = []
    train_labels = []
    test_data = []
    test_labels = []
    # kf.split returns tuple: ([all indices of training data], [all indices of test data])
    for tup in  kf.split(data_list):
        train_data.append([data_list[i] for i in tup[0]]) 
        train_labels.append([target_list[i] for i in tup[0]]) 
        test_data.append([data_list[i] for i in tup[1]]) 
        test_labels.append([target_list[i] for i in tup[1]]) 
    return train_data, train_labels, test_data, test_labels


