# CS7CS4-Machine-Learning-Kaggle-Competetion
This repository contains the code used in the in-class individual machine learning competetion for the CS7CS4 module.

We were given a dataset with numerous features: Year of Record, Age, Country, Profession, Gender, University Degree, Wears Glasses, Hair Color, Size of City, Body Height [cm] and the target label was the 'Income in EUR'
The problem was to train machine learning models on the training data provided and make predictions for the income in the test data. 

The machine learning pipeline I used for this problem is described in brief below:

1) Data Exploration: Data exploration was performed using matplotlib (histograms, scatter plots, boxplot and so on). For the categorical variables 
2) Feature Engineering: The missing values in the data were replaced with the median for numerical columns (Age, Year of Record) and with a new category called 'Not specified' or 'unknown' for the categorical columns. Some columns already had these categories so missing values were assigne to them.
The categorical columns were encoded using One-Hot encoding first and it produced reasonable error. To lower the error, other encodings like Binary encoding and Hash Encoding were tried but to no success. 
Finally, Mean Encoding was applied manually which gave monumental decrease the error. This was done by replacing the the categories with the mean of Income for rows corresponding to each category in the training set.
3) Machine Learning models: I started about by applying Linear regression and Ridge Regression which gave errors close to 80,000. Decrease in error was achieved by applying XGBoost Regression, a library for gradient boosted decision tree algorithm. 
4) Cross Validation and Grid Search: To evaluate the XGBoost model, cross validation was used to get the mean of error and standard deviation across 5 folds of training and validation data. It was observed that models with lower standard deviation and slightly more error performed better on the training set. K-Fold cross validation was significant in observing the amount of overfitting in the model.
5) Grid Search: To find the best parameters for XGBoost, the XGBoost cross validation function was used to get the optimum number of boosting rounds and sklearn's Grid Search was applied iteratively for different sets of parameters to get the optimum set of parameters. On the local machine, this improved the error down to 55,000 from 60,000. It also reduced the error on the testing set.
6) Further optimization: This was done by backward elimination. It was found that removing features like Gender and Hair Colour improved performance but not by a lot.
