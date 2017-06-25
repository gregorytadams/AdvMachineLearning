# Greg Adams
# Similar to my magic loop, a grid-search pipeline specifically for text classifiers.  

from sklearn import svm
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV 
from sklearn.pipeline import Pipeline
from os import walk
from string import digits, punctuation
import pandas as pd
from copy import deepcopy
print("Finished importing")

# Make this true to (1) make the code run in any sort of manageable timeframe and (2) not run your computer super hard.
IM_JUST_HERE_TO_TEST = False 

K = 3 
SENTIMENT_CATEGORIES = ['passed', 'not_passed']
PL = [('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), None]

CLFS = {'NB': ('NB', MultinomialNB()), \
        'SGD': ('SGD', SGDClassifier()), \
        'SVC': ('SVC', svm.SVC())
}

# Actual search, takes forever (I had to split it up and run each chunk overnight, even parallelized on 8 cores in CSIL)

CLF_PARAMS = {'NB': {'NB__alpha': (10, 1, 1e-1, 1e-2, 1e-3, 1e-4)}, \
              'SGD': { 'SGD__loss': ['hinge','log','perceptron'], 'SGD__penalty': ['l2','l1','elasticnet'], 'SGD__class_weight': ['balanced', None]}, \
              'SVC': {'SVC__C' :[0.00001, 0.0001, 0.001,0.01,0.1,1,10], 'SVC__kernel':['linear', 'poly', 'rbf', 'sigmoid']}
            }

BASIC_PARAMS = {'vect__ngram_range': [(1, 1), (1, 2), (1,3)], 'vect__binary': (True, False), \
                'tfidf__use_idf': (True, False), 'tfidf__norm': ('l1', 'l2', None), 'tfidf__smooth_idf': (True, False), \
                'tfidf__sublinear_tf': (True, False)}

# For TESTS; takes ~20 mins on CSIL computers if IM_JUST_HERE_TO_TEST = True

# CLF_PARAMS = {'NB': {'NB__alpha': (1, 0.1)}, \
#               'SGD': { 'SGD__loss': ['hinge'], 'SGD__class_weight': ['balanced', None]}, \
#               'SVM': {'SVM__C' :[0.00001, 10]}
#             }

# BASIC_PARAMS = {'vect__ngram_range': [(1, 1), (1,3)],  \
#                 'tfidf__use_idf': (True, False)} 

def remove_junk(string):
    '''
    Cleans up the text of the bills.  I also did some command-line cleaning of some of the wonkier bits
    when I converted them to text.
    '''
    string = string.lower()
    remove_digits = string.maketrans('', '', digits)
    remove_punctuation = string.maketrans('', '', punctuation)
    remove_newlines = string.maketrans('', '', '\n')
    res = string.translate(remove_digits).translate(remove_punctuation).translate(remove_newlines)
    return res


def fetch_and_format():
    '''
    Fetches and formats the data. Assumes polarity data was extracted in the current folder.  
    Hardcoded for file structure.

    Outputs:
    data, a list of raw strings from the text files
    labels, a list of strings of the SENTIMENT_CATEGORIES, corresponding to the data's labels
    '''
    print("Fetching data...")
    data = []
    labels = []
    counter = 0
    for cat in SENTIMENT_CATEGORIES:
        filenames = next(walk('./all_bills/{}/txts'.format(cat)))[2]
        for f in filenames:
            with open('./all_bills/{}/txts/{}'.format(cat, f)) as txt_f:
                data.append(remove_junk(txt_f.read()))
                labels.append(cat)
            counter += 1
            if counter % 100 == 0:
                print("{}".format(counter))
    return data, labels


def define_parameters(pl = PL, clfs = CLFS, clf_params = CLF_PARAMS, basic_params = BASIC_PARAMS):
    '''
    Generates the appropriate pipeline/parameter pairs

    inputs:
    pl, pipeline sans classifier
    clfs, dict of classifier tuples for pipeline
    clf_params, classifier parameters to search through
    basic_params, non-classifier parameters to search through

    outputs:
    list_of_pairs, list of tuples [(pipeline, parameters), ...]
    '''
    list_of_pairs = []
    for model in CLFS:
        pl_to_use = deepcopy(pl)
        pl_to_use[2] = CLFS[model]
        new_params = deepcopy(basic_params)
        new_params.update(clf_params[model]) 
        plc = Pipeline(pl_to_use)
        list_of_pairs.append((plc, new_params))
        print("Testing {} parameter combinations for model {}".format(len(list_of_pairs), model))
    return list_of_pairs


def build_and_validate(data, labels, just_testing = IM_JUST_HERE_TO_TEST):
    '''
    Builds and tests the models for each of the classifiers and parameters.
    Parallelized on all available cores.

    Inputs:
    data, labels; from fetch_and_format
    just_testing, a bool indicating whether you want real results, or want to not break your computer 

    Outputs the model evaulation report to a csv.
    '''
    pairs = define_parameters()
    for i, tup in enumerate(pairs):
        gs_clf = GridSearchCV(tup[0], tup[1], n_jobs = -1, cv = K) # -1 parallelizes on all available cores
        if just_testing:
            gs_clf = gs_clf.fit(data[:100], labels[:100])
        else:
            gs_clf = gs_clf.fit(data, labels)
        df = pd.DataFrame(gs_clf.cv_results_)
        df.to_csv('test_output_{}.csv'.format(i))
        gs_clf = None 


def go():
    '''
    Master function.
    '''
    data, labels = fetch_and_format()
    build_and_validate(data, labels)
    

if __name__ == "__main__":
    go()




