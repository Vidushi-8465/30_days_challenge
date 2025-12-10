# Day 2 – House Price Prediction

This project predicts the price of houses based of few 13 features like:
**Area,number of rooms ,tax ,crime rate, location, etc.**
The dataset we are using in this project is the Boston housing dataset

##  What This Project Does
* Loads the data
* trains 2 ML models
* Compares the accuracy of both the models
* Chooses the Best model i.e **(Random Forest)**
* Saves it as model.pkl
---

##  Algorithms Used 
### **1. Linear Regression**
It is a Simple model
It draw the best straight line that fits the data
### **Random forest Regressor**
It is a better and a bit complex model
It Creates many decision trees --> who then give their guesses --> and then it averages them.
It is thus called as **Team of smart Trees**
---

##  Output
* Linear Regression Performance:
(0.5332001304956555, 0.5558915986952442, np.float64(0.7455813830127763), 0.575787706032451)
* MAE = 0.53, MSE = 0.55, RMSE= 0.74, R2=0.57

* Random Forest Performance:
(0.32681185043604677, 0.2539759249192041, np.float64(0.5039602414072009), 0.8061857564039718)
* MAE=0.32 , MSE= 0.25 ,RMSE= 0.50 , R2= 0.80
* Thus Random Forest performs better and gives higher Accuracy 

* Random forest Model is thus saved for future prediction.
---

##  What I Learned
* Regression models
* Error metrics (MAE, MSE, RMSE, R²)
* Saving trained models
* Basic ML workflow
---



