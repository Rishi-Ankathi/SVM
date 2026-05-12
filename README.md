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
        
