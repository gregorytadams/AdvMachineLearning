README.txt

Problem 1 
Handwritten in three parts: Problem_1a.pdf, Problem_1b.pdf, Problem_1c.pdf.
The names do not correspond to the problem part, but they are consecutive.

Problem 2
python3 Problem_2.py
The stationary distribution of the random surfer prints to the terminal.

Problem 3
python3 Problem_3.py
Prints the f1 scores (average of cross-validated F1 scores) of both models, as
well as the bootstrapped p-values.  To get more evaluation statistics of the models, set
VERBOSE variable (global) to True.

A few things to consider in the models from Problem 3:
(1) the counts models tend to have a higher F1 score.  It seems that the TFIDF
models have a higher precision, but lower recall.  This may be driven by the 
