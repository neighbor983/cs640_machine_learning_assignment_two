# cs640_machine_learning_assignment_two

## Assignment 2
### Due on September 7, 2018
1.	You are given 15 well distributed training pattern of dimensionality 4.  
    a.	What is the probability that a linear classifier separates a randomly selected dichotomy of training patterns? 
    b.	What is the minimum degree polynomial function that guarantees separation of any dichotomy of 15 sample patterns?

2.	You are given training patterns {X1,, .  .  .  .  , XN}  from two classes. All patterns are augmented by 1 and patterns of Class-2 are multiplied by -1 as discussed in the class. 

The cost function J(θ) = ∑I = 1 to N (1/Xi tXi) [|θtXi|2-  |θtXi|θtXi] attains its only minimum value of zero if θtXi  > 0 for all Xi. 

Develop a machine learning algorithm based on gradient descent approach and the above cost function to learn θ from training data to separate the two classes. 

3.	You are given the following training data.

Class1: (2  4)t, (3  3)t 

Class2: (6 12)t, (8  10)t

Starting with an initial parameter vector [0 1 1]t

    a.	Illustrate 4 iterations of perceptron – formulation 1.
    b.	Illustrate 4 iterations of perceptron – formulation 2.
    c.	Illustrate 4 iterations of relaxation with b = 1.
    d.	Illustrate 4 iterations of Ho-Kashyap algorithm with b = [1 1 1 1]t. 
