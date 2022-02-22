# Insurance_Premium_Prediction

ML MODEL DEPLOYMENT USING HEROKU

link : https://medicalpremiumpredict.herokuapp.com/

Problem Statement :

The goal of this project is to give people an estimate of how much they need based on their individual health situation then they can work with any health insurance carrier and its plans and perks while keeping the estimated cost in mind. This can assist a person in concentrating on the health side of an insurance policy rather than the ineffective part.

Dataset :
The dataset is taken from a Kaggle.

Approach :
Applying machine learing tasks like Data Exploration, Data Cleaning, Feature Engineering, Model Building and model testing to build a solution that should able to predict the premium of the person for health insurance.

Data Exploration : Exploring the dataset using pandas, numpy, matplotlib and seaborn.
Exploratory Data Analysis : Plotted different graphs to get more insights about dependent and independent variables/features.
Feature Engineering : Numerical features scaled down and Categorical features encoded.
Model Building : In this step, first dataset Splitting is done. After that model is trained on different Machine Learning Algorithms such as:
1) Linear Regression
2) Decision Tree Regressor
3) Random Forest Regressor
4) Gradient Boosting Regressor
5) Support Vector Reressor

Model Selection : 
Tested all the models to check the RMSE, R-squared and Cross Validation Score.

Pickle File : Selected model as per best RMSE score, R-squared, Cross Validation Score and created pickle file using pickle library.

Webpage and Deployment : Created a web application that takes all the necessary inputs from the user & shows the output. Then deployed project on the Heroku Platform.
