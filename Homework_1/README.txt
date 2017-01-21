README
All problems for answered below.


Q1. Suppose you are building an n-gram word model and your input is alist of sentences.  Argue why you should add (n-1) pseudo start words at the beginning of each sentence, and one pseudo end word at the end of each sentence.

Adding in pseudo start/end words allows you to take into account the grams at the beginning and end of each sentence more equally.  Without the pseudo start words, the first 3-gram, for example, would be the first three words in the sentence, followed by words 2-4.  Later words would get counted three times: once as the first, second, and third words of the 3-gram.  By adding in the pseudo words, each actual word gets counted 3 times, and the pseudo-words get deemphasized.  Additionally, the start and end words allow the model to assign a probability to the sentence ending or, conversely, assign different probabilities to how the sentence starts.

Q2. Hand-written in Problem_2.pdf

P_hat(Sam|am) = lambda_2 * P(Sam|am) + lambda_1 * P(Sam)
P(Sam) = 4/25 = .16
P(Sam|am) = 2/3 = .66
With lambda_2 = 0.5 and lambda_1 = 0.5 (equal weights):
P_hat(Sam|am) = 0.5 * 0.66 + 0.5 * 0.16 = 0.4133
 
Q3. > python3 Problem_3.py



