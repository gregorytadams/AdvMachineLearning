# my_library.py


from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import KFold
from sklearn import preprocessing
import numpy as np
import pandas as pd 


def get_model_evaluation_stats(y_true, y_pred, labels=None, pos_label=1, average='binary', \
                               sample_weight=None, title="Summary", print_scores=False):
    '''
    Custom summary function to get a quick look at my models.

    Requires import above: 'from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score'
    
    Inputs:
    (all) same inputs as sklearn.model_selection inputs of the same name.  Refer to sklearn documentation at 
    http://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics
    '''
    f1 = f1_score(y_true, y_pred, pos_label=pos_label, average=average, sample_weight=sample_weight)
    precision = precision_score(y_true, y_pred, pos_label=pos_label, average=average, sample_weight=sample_weight)
    recall = recall_score(y_true, y_pred, pos_label=pos_label, average=average, sample_weight=sample_weight)
    acc = accuracy_score(y_true, y_pred, sample_weight=sample_weight)
    if print_scores:
        print("TITLE: {}".format(title))
        print("F1: {}".format(f1))
        print("Precision: {}".format(precision))
        print("Recall: {}".format(recall))
        print("Accuracy: {}".format(acc))
        print("-"*30)
    return [f1, precision, recall, acc]

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


def numberize(df, col):
    '''
    replaces categories with numbers
    http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html#sklearn.preprocessing.LabelEncoder

    inputs: 
    df, a pandas DataFrame
    col, the column to numberize

    outputs:
    df, the DtataFrame with col numberized
    '''
    le = preprocessing.LabelEncoder()
    df[col] = le.fit_transform(list(df[col].values))
    return df

def split_to_indicators(df, col):
    '''
    Grabs categorical variables and makes them dummies for the models.
    '''
    df2 = pd.get_dummies(df[col])
    df = pd.concat([df, df2], axis=1)
    df = df.drop(col, 1)
    return df



# def format_for_models(data, response, predictor):
#     y = data[response].values
#     if type(predictor) == str: #if there's only one predictor
#         x = data[predictor].values
#         x = x.reshape(len(x), 1)
#         return x, y
#     else: #if it's a list of things
#         my_array = data[predictor.pop(0)].values
#         my_array = my_array.reshape(len(my_array), 1)
#         for i in predictor:
#             a = data[i].values
#             a = a.reshape(len(a), 1)
#             my_array = np.concatenate((my_array, a), axis=1)
#         return my_array, y
