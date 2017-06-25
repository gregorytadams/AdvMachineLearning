README.txt
Homework 3

Problem 1

def split_words(string):
    if string = '':
        return string 
    rv = []
    split_index = 0
    max_qual = 0
    for index in range(len(string)): 
        qual = quality(string[:index])
        if qual > max_qual:
            max_qual = qual
            split_index = index
    rv.append(string[:split_index])
    rv += split_words(string[index:])
    return rv

a = 2 because the algorithm is just less than O(n^2) in the worst case (e.g. string = "IIIIIIIIIIIII").  O(n) is the best case (e.g. string = "hello").
For efficiency's sake, you can minimize calls by replacing len(string) with the maximum length of a word you're likely to see.
This algorithm assumes that longer words have a higher quality than shorter words (e.g. "meet" has a higher quality than "me"). 


Problem 2

Formulas:

    HMM probabilities:
    (i) P(pronoun|start) * P(I|pronoun) * P(verb|pronoun) * P(eat|verb) * P(end|verb)
    (ii) P(pronoun|start) * P(I|pronoun) * P(pronoun|pronoun) * P(eat|pronoun) * P(end|pronoun)

    Church probabilities:
    (i) P(pronoun|start) * P(pronoun|I) * P(verb|pronoun) * P(verb|eat) * P(end|verb)
    (ii) P(pronoun|start) * P(pronoun|I) * P(pronoun|pronoun) * P(pronoun|eat) * P(end|pronoun)

    HMM calculations:
    (i) 1 * (x/(x+y)) * b * (v/(u+v)) * 1
    (ii) 1 * (x/(x+y)) * a * (y/(y+x)) * c

    Church calculations:
    (i) 1 * (x/(x+u)) * b * (v/(y+v)) * 1
    (ii) 1 * (x/(x+u)) * a * (y/(y+v)) * c

a) 
    HMM gets it right; Church gets it wrong, i.e.:
    For HMM, P(i) > P(ii)
    For Church, P(i) < P(ii)

    u = 0   v = 1   x = 1   y = 1
    a = 0.44    b = 0.12    c = 0.14

    HMM:
    P(i) = 1 * (1/(1+1)) * 0.12 * (1/(0+1)) * 1 = 0.5 * 0.12 = 0.06
    P(ii) = 1 * (1/(1+1)) * 0.44 * (1/(1+1)) * 0.44 = 0.44^2 * 0.5^2 = 0.0484
    P(i) > P(ii)

    Church:
    P(i) = 1 * (1/(1+0)) * 0.12 * (1/(1+1)) * 1 = 0.5 * 0.12 = 0.06
    P(ii) = 1 * (1/(1+0)) * 0.44 * (1/(1+1)) * 0.44 = 0.44^2 * 0.5 = 0.0968
    P(i) < P(ii)

b)
    HMM gets it wrong; Church gets it right, i.e.:
    For HMM, P(i) < P(ii)
    For Church, P(i) > P(ii)

    v = 100     u = 100     x = 1   y = 100
    a = 0.4     b = 0.2     c = 0.4

    HMM:
    P(i) = 1 * (1/(1+100)) * 0.2 * (100/(100+100)) * 1 = 0.00099
    P(ii) = 1 * (1/(1+100)) * 0.4 * (100/(100+1)) * 0.4 = 0.4^2 * 0.01 = 0.0016
    P(i) < P(ii)

    Church:
    P(i) = 1 * (1/(1+100)) * 0.2 * (100/(100+100)) * 1 = ~0.001
    P(ii) = 1 * (1/(1+100)) * 0.4 * (100/(100+100)) * 0.44 = 0.00087
    P(i) > P(ii)

Problem 3: HMM code in HMM.py.  I was not able to finish the forward-backward algorithm.
