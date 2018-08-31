'''
1.	You are given 15 well distributed training pattern of dimensionality 4.  
a.	What is the probability that a linear classifier separates a randomly selected dichotomy of training patterns? 
b.	What is the minimum degree polynomial function that guarantees separation of any dichotomy of 15 sample patterns?


Notes
Dichotomization Capacity
1. In theory, there are 2^N different ways of classifying N well distributed 
sample patterns into two classes. N samples are well distributed in 
n-dimensional patter space, if there exist no subset of (n + 1) patterns that 
lies on a (n – 1)-dimensional hyperplane. For example, for n = 2, no subset of 3 
patterns must be collinear. 
2. Usually, a linear decision function is not able to implement all dichotomies. 
3. It has been shown that the number of dichotomies (D(N, n)) a linear decision 
function can implement is given by 2∑k = 0 to n (N – 1)! / ((N - 1 -k)! k!), for 
N > ( n+1). If N ≤ (n + 1) then it is 2N. 
4. If we plot D(N, n)/2Nas a function of N/(n + 1), all curves pass through 
(2, 0.5). The steepness of the curve increases as n increases giving a 
thresholding effect for large values of n. This means that we are almost certain 
to implement a random dichotomy using a linear decision function if 
N < 2(n + 1). The probability of implementing a given dichotomy sharply 
decreases if N ≥ 2(n + 1). For this reason, 2(n + 1) is called the 
dichotomization capacity of the decision function. 
'''
Number_of_samples = 15;
dimensionality = 4;

probability = Number_of_samples / ( dimensionality + 1 )
print(probability);