# Problem_3_classifier.py
# Gregory Adams

from math import log
from operator import itemgetter

class NGramClassifier(object):

    def __init__(self, lambda1, lambda2, lambda3):
        '''
        Tnitializes the classifier.  Stores the lambdas as variables and initializes the models as an empty dictionary
        
        Inputs:
        lambda1, lambda2, lambda3: floats between 0 and 1.  Must sum to 1.
        '''
        self.lambdas = [lambda1, lambda2, lambda3]
        self.models = {}

    def train(self, list_of_sentences, gen):
        '''
        Trains the model on list of sentences.  
        Creates a subdictionary in models with the letter counts.

        inputs:
        list_of_sentences, a list of lists in which each sublist contains a tokenized sentence
        gen, a genre as a string
        '''
        self.models[gen] = { "uni":{ "<unk>":0, "sum":0 }, "bi":{ "<unk>":0, "sum":0 }, \
                            "tri":{ "<unk>":0, "sum":0 } }
        for sentence in list_of_sentences:
            tokens = ["<s0>", "<s1>"] + sentence + ["<e0>", "<e1>"]
            for index, word in enumerate(tokens[2:]):
                self._add_gram("uni", word, gen)
                self._add_gram("bi", (word, tokens[index - 1]), gen)
                self._add_gram("tri", (word, tokens[index - 1], tokens[index - 2]), gen)

    def _add_gram(self, gram_type, gram, gen):
        '''
        Adds one gram to the model

        inputs:
        gram_type: either "uni", "bi", or "tri"
        gram: the specific sequence of letterns/words to be added as a tuple
        gen: the genre
        '''
        if gram in self.models[gen][gram_type]:
            self.models[gen][gram_type][gram] += 1
        else: 
            self.models[gen][gram_type][gram] = 1
            self.models[gen][gram_type]["<unk>"] += 1
        self.models[gen][gram_type]["sum"] += 1

    def predict(self, list_of_sentences):
        '''
        Returns the predicted genre for each sentence, in order.

        inputs:
        list_of_sentences, a list of lists in which each sublist is a tokenized sentence.  Each sentence will be labelled. 
        '''
        rv = []
        for sentence in list_of_sentences:
            probs = []
            tokens = ["<s0>", "<s1>"] + sentence + ["<e0>", "<e1>"]
            for gen in self.models: 
                probs.append((gen, self._get_prob(tokens, gen)))
            rv.append(max(probs,key=lambda item:item[1])[0])
        return rv

    def _get_prob(self, tokens, gen):
        '''
        Gets the probability that a given sentence is in a specified genre.

        inputs:
        tokens, a tokenized sentence as a list
        gen, a genre as a string
        '''
        prob = 0
        for index, word in enumerate(tokens[2:]):
            bigram = (word, tokens[index - 1])
            trigram = (word, tokens[index - 1], tokens[index - 2])
            P_tri = log(self.models[gen]["tri"].get(trigram, self.models[gen]["tri"]["<unk>"]) / \
                    self.models[gen]["bi"].get(bigram, self.models[gen]["bi"]["<unk>"]))
            P_bi = log(self.models[gen]["bi"].get(bigram, self.models[gen]["bi"]["<unk>"]) / \
                    self.models[gen]["uni"].get(bigram, self.models[gen]["uni"]["<unk>"]))
            P_uni = log(self.models[gen]["uni"].get(bigram, self.models[gen]["uni"]["<unk>"]) / \
                    self.models[gen]["uni"]["sum"])
            prob += self.lambdas[2] * P_tri + self.lambdas[1] * P_bi + self.lambdas[0] * P_uni
        return prob

