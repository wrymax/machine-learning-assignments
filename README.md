# machine-learning-assignments

This repository includes the assignments of Machine Learning course in Cousera, mainly implemented in Matlab. 

All of them obtains score 95 - 100.

## Week 2. Linear Regression

Week 2 assignments mainly works on linear regression algorithms, which leverage 「Gradient Descent」we implement ourselves in the code.

## Week 3. Logistic Regression

Week 3 assignments mainly works on logistic regression algorithms. Instead of implementing the gradient descent function myself, I leverage「fminunc」method in Matlab, which gives an efficient implementation for gradient descent.

## Week 4. Multi-Class classification and Neural Networks

In this week, we approached the exiciting「Neural Network」to implement classification algorithms. In the code, we implement a multi-class classification  algorithm and One-vs-All algotithm.

## Week 5. Neural Networks Learning

To enhance and optimise of Neural Network, we introduce「Feed Forward」to compute the prediction of traning set X, based on which we can compute cost function J of  theta.

Also we leverage「Back Propagation」method to compute Gradient of J(theta). To verify it, we should use「Gradient Checking」techniques, which comes for numeric estimate, to make sure that the gradient is accurate.

To put it together, the code in ex4.m shows complete process of training a Neural Network:

- Randomly initialize weights;
- Implement forward-propagation to get h(theta);
- Implement code to compute cost function J(theta);
- Implement back-propagation to compute partial derivative of J(theta), which is the gradient used for gradient descent algorithm;
- Regularize the gradient of J(theta);
- Start training by fmincg.

## Week 6. Neural Networks Practical Stuff

This week, we implement some practical code for neural network.

## Week 7. Support Vector Machine

Here simply implements an spam email classifier by SVM.
