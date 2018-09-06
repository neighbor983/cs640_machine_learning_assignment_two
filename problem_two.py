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