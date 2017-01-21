# Problem_3.py 
# Gregory Adams
# Runs everything necessary for Problem 3

print("Fetching Corpus...")
from nltk.corpus import brown
from Problem_3_classifier import NGramClassifier
from sklearn.model_selection import  KFold
from sklearn.metrics import f1_score
import numpy as np
from operator import itemgetter
print("Building Models...")

TEST_GENRES = ["mystery", "romance", "science_fiction"]
LAMBDAS_TO_TRY = [[0, 0, 1], [0.1, 0.2, 0.7], [0.33, 0.33, 0.34], [0, 0.5, 0.5]]
K = 5

def get_data(list_of_genres):
    '''
    Takes a list of genres and grabs the sentences from the Brown corpus

    input: list of strings (genres)

    output: dictionary with genre keys and list of sentences values
    '''
    data = {}
    for genre in list_of_genres:
        data[genre] = list(brown.sents(categories=genre))
    return data

def format_data(data):
    '''
    Takes data from get_data and gives a list of sentences, plus another list with their corresponding genres

    inputs: data, from get_data

    outputs: 
    data_list, a list of lists in which each sublist is a tokenized sentence'
    target_list, a list of the labels for each sentence in data_list
    '''
    target_list = []
    data_list = []
    for genre in data:
        target_list += [genre]*len(data[genre])
        data_list += data[genre]
    return data_list, target_list

def train_model(classifier, X_train, Y_train):
    '''
    Trains the model for each genre
    '''
    for genre in TEST_GENRES:
        #List comprehension just grabs the sentences from each genre; train_test_split messed them around
        list_of_sentences = [X_train[index] for index, item in enumerate(Y_train) if item == genre] 
        classifier.train(list_of_sentences, genre)

def get_k_folds(data_list, target_list, k):
    '''
    Gets the k folds for cross validation

    Inputs:
    data_list, a list of lists, each sublist of which is a sentence
    target_list, a list of strings, each element of which is the label for data_list

    Outputs:
    (all), a list of k folds, each of which is a list of lists with each sublist containing a sentence
    '''
    kf = KFold(n_splits=K, shuffle=True, random_state=0)
    train_data = []
    train_labels = []
    test_data = []
    test_labels = []
    for tup in  kf.split(data_list):
        train_data.append([data_list[i] for i in tup[0]]) 
        train_labels.append([target_list[i] for i in tup[0]]) 
        test_data.append([data_list[i] for i in tup[1]]) 
        test_labels.append([target_list[i] for i in tup[1]]) 
    return train_data, train_labels, test_data, test_labels

def get_predictions(classifier, train_data, train_labels, test_data, test_labels):
    '''
    Takes the data folds from get_k_folds and trains the classifier to get the F1 scores for each fold
    
    inputs:
    classifier, an NGramClassifier object
    (all others), lists from get_k_folds

    outputs:
    F1_scores, a list of floats, each being an f1 score
    '''
    f1_scores = []
    for i in range(len(train_data)):
        train_model(classifier, train_data[i], train_labels[i])
        pred_labels = classifier.predict(test_data[i])
        f1_scores.append(f1_score(test_labels[i], pred_labels, labels=TEST_GENRES, average='micro'))
    return f1_scores

def go():
    '''
    Master function.  Runs entire algorithm.
    '''
    data_list, target_list = format_data(get_data(TEST_GENRES))
    final_f1s = []
    for LAMBDAS in LAMBDAS_TO_TRY:
        classifier = NGramClassifier(LAMBDAS[0], LAMBDAS[1], LAMBDAS[2])
        train_data, train_labels, test_data, test_labels = get_k_folds(data_list, target_list, K)
        f1s = get_predictions(classifier, train_data, train_labels, test_data, test_labels)
        final_f1s.append((LAMBDAS, sum(f1s) / len(f1s))) # micro- and macro- averaging should have the same result when each fold is of equal size
    return max(final_f1s,key=lambda item:item[1])

if __name__ == "__main__":
    lambdas, f1 = go()
    print("Genres: {}:".format(TEST_GENRES))
    print("Lambdas: {}".format(lambdas))
    print("{}-fold cross-validated F1 score: {}".format(K, f1))