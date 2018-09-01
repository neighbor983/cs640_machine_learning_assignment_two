import math

sample_patterns = 15;
dimensionality = 4;

def dichotomies(sample_patterns, dimensionality):
    '''
    description:
        determines the number dichotomies given sample_patterns and dimensionality
    params:
        sample_patterns = number
        dimensionality = number
    output:
        number
    '''
    total = 0;
    numerator = math.factorial( sample_patterns - 1 );
    for n in range( dimensionality + 1):
        total += numerator / ( math.factorial(sample_patterns - 1.0 - n) * math.factorial( n ) );
    return total;
    
def dichotomization_capacity(dimensionality):
    '''
    description:
        determines the dichotomization capacity given the dimensionality
    params:
        dimensionality = number
    output:
        number
    '''
    return 2 * ( dimensionality + 1 );

print( dichotomies(sample_patterns, dimensionality) / (2**sample_patterns));
print(dichotomization_capacity(dimensionality));
