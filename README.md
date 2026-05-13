# This repository consists of SVC and SVR support vector machines implementation with a deployed streamlit application code that predicts loan approval

# svm

1. why we need svm?
   - svm is mainly used for

     . complex data suppression

     . classification problems

     . binary classification

   - ex: spam vs not spam

     cat vs dog

     disease vs no disease

3. Problem with logistic regression

   - logistic regression works well when,
     . data is linearly separable

   - if the data is complex

   - the classes we predict overlap

   - decision boundary is not straight
     This is where svm becomes powerfull


-> **Parameters** are the values learned from the data

-> **Hyper parameters** are the values which are not learnt from the data and are not related to the data , they are the input parameters which we  pass as input to the model

-> **difference btw grid search cv , randomised search cv**
        GridSearchCV performs an exhaustive, predefined search, while RandomizedSearchCV samples values randomly based on a budget
        

# Equation of hyperplane
 
  w.x + b =0

  where, w= weights
         x= input
         b= bias/intercept

  Prediction rule:
    if w.x + b >= 0 --> class 1
    else --> class 2

# Hyper parameters

linear = linear data

poly = polynomial relationship

rdf = non-linear complex data

sigmoid = neural-network-like behaviour


c = 1  will control overfitting and underfitting

small c --> wider margin --> underfit

large c --> tries to classify all points --> may overfit


gamma = scale

controls how far the influence of one data point reaches 

small gamma value --> generalised model

high gamma value --> model memorizes data
