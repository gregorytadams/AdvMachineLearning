README.txt

Problem 1 
Problem_1.pdf. 3 pages.

Problem 2
python3 Problem_2.py
The stationary distribution of the random surfer prints to the terminal.

Problem 3
python3 Problem_3.py
Prints the f1 scores (average of cross-validated F1 scores) of both models, as well as the bootstrapped p-values.  To get
more evaluation statistics of the models, set VERBOSE variable (global) to True.

A few things to consider in the models from Problem 3:
(1) the counts models tend to have a higher F1 score.  It seems that the TFIDF models have a higher precision, but lower
recall, so are more conservative.
(2) The script is very slow.  A big part of that is driven by the data (just importing functions and fetching the data
takes a while), but so is the model building/validation.  One way to improve this would be to reduce K, but that would
also have the effect of weakinening the strength of my p_value.
(3) It may be more useful, depending on what I would hypothetically be using this model for, to compare the models based
not juts on F1, but on precision/recall individually.  It would give me more control over where the model is making
mistakes.
(4) It would probably help to experiment with different parameters in my model to improve predictive strength.  Changing
things like the smoothing parameter and seeing what works best would make the predictions less arbitrary.

