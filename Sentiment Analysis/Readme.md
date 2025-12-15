### Day 6 (Project 2): Sentiment Analysis using NLP

This project analyzes text reviews and predicts whether the sentiment is
positive or negative.

## Overview:

Sentiment Analysis helps understand user opinions from text data.
The model converts text into numbers and classifies sentiment.

### Dataset used:
Text reviews dataset with sentiment labels

### Algorithms used:

* 1. TF-IDF Vectorizer
   * What is it?
> Converts text into numerical form
> Gives importance to meaningful words

   * Why TF-IDF?
> Removes common words
> Highlights important words

* 2. Logistic Regression
> A simple and effective classification algorithm
> Used for binary sentiment classification

### Using Logistic regression was fail due to few factors: ( accuracy =0.0)
1. comparetively small dataset ( we require min 100 lines for the model to predict properly)
2. there are no patterns which model can't generalise
3. There is a class imbalance i.e only class = "Positive" is seen whereas class = "negative" and class ="Neutral" are not seen 


### So in order to fix this we can : 
1. Either create a big dataset 
2. Change the model from logistic regression to Naive Bayes
3. or use stratify =y in train and test model , so that each class or sentiment is seen
4. we can also have binary sentiment i.e only positive and negative rather than multiclass i.e positive ,negative, and neutral
5. or we can just supress the warning rather than fixing it :)


### So i have create a another ipynb file with naive bayes as the model
after changing the model 
the accuracy returned as 0.25 
thus showing that yes the csv file was read properly and all the classes were seen