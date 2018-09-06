'''
3.	You are given the following training data.

Class1: (2  4)t, (3  3)t 

Class2: (6 12)t, (8  10)t

Starting with an initial parameter vector [0 1 1]t

    a.	Illustrate 4 iterations of perceptron – formulation 1.
    b.	Illustrate 4 iterations of perceptron – formulation 2.
    c.	Illustrate 4 iterations of relaxation with b = 1.
    d.	Illustrate 4 iterations of Ho-Kashyap algorithm with b = [1 1 1 1]t. 
    
    
Perceptron
Perceptron is a linear classifier.  It learns the decision function from 
training data using gradient descent approach.  There are many formulation of 
the same basic algorithm. Given: We are given training patterns 
{X1,, .  .  .  .  , XN} from both classes.  
Let n denote the dimensionality of X and N denote the number of training patterns. For each training pattern, we know the class membership as illustrated by the following example. Class1: (72  220)t, (74  200)tClass2: (60 120)t, (63  110)t
Usually, we augment each pattern with a 1 as its first component as shown below.Class1: (1  72  220)t, (1  74  200)tClass2: (1  60 120)t, (1  63  110)tFormulation 1: The goal is to find θt= [θ0 , θ1, .  .  ., θn] such that θtX is greater than zero if X belongs to Class1, and θtX is less than zero if X belongs to Class2.  Mathematically, we have to solve a set of N greater than zero and less than zero inequalities.The cost function J(θ) = ∑Class1 (|θtX| - θtX)  + ∑Class2 (|θtX| + θtX) is greater than or equal to zero, and achieves its minimum value of zero if θtX is greater than zero if X belongs to Class1 and θtX is less than zero if X belongs to Class2.  For this case, the gradient descent algorithm is as follows.Step 1: Select an initial θ(1) randomly.Step 2:  During iteration k, present X(k) and compute θt(k)X(k).Step 3:  If θt(k)X(k) > 0 and X(k) belongs to Class1 θ(k+1) = θ(k)else if θt(k)X(k) ≤  0 and X(k) belongs to Class2 θ(k+1) = θ(k)else if θt(k)X(k) > 0 and X(k) belongs to Class2 θ(k+1) = θ(k) – α X(k)else if θt(k)X(k) ≤ 0 and X(k) belongs to Class1 θ(k+1) = θ(k) + α X(k)k = (k+1) mod NStep: If you have gone through N iterations without changing θ stop.  Otherwise, go to Step 3.Formulation 2:  Multiply augmented patterns of Class2 by -1.Class1: (1   72  220)t, (1  74  200)tClass2: (-1   -60  -120)t, (-1  -63  -110)tThe goal is to find θ such that θtX is greater than zero for all X.  In other words, mathematically, we must solve a set of N greater than zero inequalities.
Define a cost function J(θ ) = (1/2) ∑i = 1 to N (|θtXi| - θtXi) (summation over all training patterns).  The cost function J has only one minimum (zero) and the minimum is attained when θtX > 0 for all X in the training set.  Therefore, solving the set of inequalities is equivalent to minimizing J.  The gradient descent approach may be used for minimizing J.Grad[J] with respect to θ= 0 if θtX > 0, and –X if θtX ≤ 0. Therefore, perceptron algorithm that learns the decision rule from training set {X1,.  .  .  .  , XN} is as below.Step 1:  Select θ(1) randomly. Step 2:   During iteration k, present X(k)If θt(k)X(k) > 0 thenθ(k+1) = θ(k)elseθ(k+1) = θ(k) + αX(k)k = (k + 1) mod NStep 3: If all patterns are correctly classified in a row stop.  Otherwise, go to Step 2. Formulation 3: Assumes that augmented patterns of Class2 are multiplied by -1 as in Formulation 2. The cost function is defined as J(θ) = ∑misclassified X (- θtX).   The summation is over all X for which θt X ≤ 0.  If all patterns are correctly classified J(θ)  is zero. Otherwise, it is positive. The algorithm is as follows.Step 1: Select an initial θ(1) randomly. Step 2:  During iteration k, present X(k) and compute θt(k)X(k).Step 3:  If θt(k)X(k) > 0θ(k+1) = θ(k)else θ(k+1) = θ(k) + α X(k)k = (k+1) mod NStep 4: If you have gone through N iterations without changing θ stop.  Otherwise, go to Step 3.All algorithms I have given are in single-sample-mode.  All algorithms can be run in batch-mode where correction is made once after presenting all N samples.    The perceptron guarantees to find a decision boundary to separate the two classes if the classes are linearly separable. However, the solution is not optimal. The trained Perceptron assigns a new pattern X to Class1 if θt X > 0.  Otherwise, X is assigned to Class2. Handling more than two classes: We can solve the multi-class problem using one of three different approaches based on the situation. Assume that there are K classes.
Case 1: Each class is separable from the rest by a single linear decision function.Now, we can find K decision functions d1(X), d2(X), . . ., dK(X) such that di(X) is greater than zero for all patterns that belong to Classi and less than or equal to zero for patterns of other classes. Case 2: Classes are pairwise linearly separable.Now we need to find K(K-1)/2 decision functions.  A new pattern X is assigned to Classi if dij(X) is greater than zero for all j ≠ i. Case 3: It may be possible to find d1(X), d2(X), . . ., dK(X) such that if X belongs to Classi then di(X) is greater than dj(X) for all j ≠ i. The training algorithm changes slightly for this case.
'''
import numpy as np;
from numpy.linalg import inv;

initial_theta  = np.array([0, 1, 1]);
class_1 = [ np.array([2, 4]), np.array([3,  3]) ];
class_2 = [ np.array([6, 12]), np.array([8, 10]) ];

#augment with a one to all vectors
class_1 = [ np.array([1, 2, 4]), np.array([1, 3,  3]) ];
class_2 = [ np.array([1, 6, 12]), np.array([1, 8, 10]) ];
sample_set = [];

for item in class_1:
    sample_set.append(item);
    
for item in class_2:
    sample_set.append(item);

newDot = initial_theta.T.dot(sample_set[0]);

#c 
alpha = .1;
theta = np.matrix([0, 1, 1]);
X_1 = np.matrix([1, 2, 4]).T;
X_2 = np.matrix([1, 3, 3]).T;
X_3 = np.matrix([-1, -6, -12]).T;
X_4 = np.matrix([-1, -8, -10]).T;
b = 1;
X3_Distance = np.linalg.norm(X_3) ** 2;
X4_Distance = np.linalg.norm(X_4) ** 2;
diff = -.1 * ((theta * X_3 - b) / (X3_Distance)).item() * X_3;
theta_new = np.matrix([(theta.item(0,0) + diff.item(0,0)), (theta.item(0,1) + diff.item(1,0)), (theta.item(0,2) + diff.item(2,0))]);
#print(.1 * (theta * X_3 - b) / (np.linalg.norm(X_3) ** 2) * X_3);

theta = theta_new;
diff = -.1 * ((theta * X_4 - b) / (X4_Distance)).item() * X_4;
theta_new =  np.matrix([(theta.item(0,0) + diff.item(0,0)), (theta.item(0,1) + diff.item(1,0)), (theta.item(0,2) + diff.item(2,0))]);


theta = theta_new;
diff = -.1 * ((theta * X_3 - b) / (X3_Distance)).item() * X_3;
theta_new =  np.matrix([(theta.item(0,0) + diff.item(0,0)), (theta.item(0,1) + diff.item(1,0)), (theta.item(0,2) + diff.item(2,0))]);

theta = theta_new;
diff = -.1 * ((theta * X_4 - b) / (X4_Distance)).item() * X_4;
theta_new =  np.matrix([(theta.item(0,0) + diff.item(0,0)), (theta.item(0,1) + diff.item(1,0)), (theta.item(0,2) + diff.item(2,0))]);

theta = theta_new;
diff = -.1 * ((theta * X_3 - b) / (X3_Distance)).item() * X_3;
theta_new =  np.matrix([(theta.item(0,0) + diff.item(0,0)), (theta.item(0,1) + diff.item(1,0)), (theta.item(0,2) + diff.item(2,0))]);

theta = theta_new;
diff = -.1 * ((theta * X_4 - b) / (X4_Distance)).item() * X_4;
theta_new =  np.matrix([(theta.item(0,0) + diff.item(0,0)), (theta.item(0,1) + diff.item(1,0)), (theta.item(0,2) + diff.item(2,0))]);

theta = theta_new;
diff = -.1 * ((theta * X_3 - b) / (X3_Distance)).item() * X_3;
theta_new =  np.matrix([(theta.item(0,0) + diff.item(0,0)), (theta.item(0,1) + diff.item(1,0)), (theta.item(0,2) + diff.item(2,0))]);

theta = theta_new;
diff = -.1 * ((theta * X_4 - b) / (X4_Distance)).item() * X_4;
theta_new =  np.matrix([(theta.item(0,0) + diff.item(0,0)), (theta.item(0,1) + diff.item(1,0)), (theta.item(0,2) + diff.item(2,0))]);
print(theta_new);


# d. 
b = np.matrix([1, 1, 1, 1]).T
X = np.matrix([[1,2,4],[1,3,3],[-1,-6,-12],[-1,8,-10]]);

alpha =.1;
theta = np.matrix([0, 1, 1]).T;

#First Iteration
Xtheta =  X * theta;
error = Xtheta - b;
addError = error + abs(error);
b_new = b + alpha * addError;
theta_new = inv(X.T * X) * X.T * b_new;

#Second iteration
b = b_new;
theta = theta_new;
Xtheta = X * theta;
error = Xtheta - b;
addError = error + abs(error);
b_new = b + alpha * addError;
theta_new = inv(X.T * X) * X.T * b_new;

#Third Iteration
b = b_new;
theta = theta_new;
Xtheta = X * theta;
error = Xtheta - b;
addError = error + abs(error);
b_new = b + alpha * addError;
theta_new = inv(X.T * X) * X.T * b_new;

#Fourth Iteration
b = b_new;
theta = theta_new;
Xtheta = X * theta;
error = Xtheta - b;
addError = error + abs(error);
b_new = b + alpha * addError;
theta_new = inv(X.T * X) * X.T * b_new;


