'''
2. You are given training patterns {X1,, .  .  .  .  , XN}  from two classes. 
All patterns are augmented by 1 and patterns of Class-2 are multiplied by -1 as 
discussed in the class. 

The cost function J(θ) = ∑I = 1 to N (1/Xi tXi) [|θtXi|2-|θtXi|θtXi] attains 
its only minimum value of zero if θtXi  > 0 for all Xi. 

Develop a machine learning algorithm based on gradient descent approach and the 
above cost function to learn θ from training data to separate the two classes. 
'''
import numpy as np;

def algorithm(training_patterns, α):
    '''
    description:
        Returns a theta if the the training_patterns provided are linearly 
        separable by the two classes.
    inputs:
        training_patterns = np.matrix
        α = number
    output:
        vector
    '''

    theta = np.ones(len(training_patterns[0]));
    changeFlag = True;
    while(changeFlag):
        changeFlag = False;
        for data in training_patterns:
            if(theta.T * data < 0):
                theta = theta + α * data;
                changeFlag = True;
    return theta;        