### day : Image emotion Detector(IED)

* 1. What am I building: 
>IED that takes a face image
>uses transfer Learning(MobileNetv2)
> and then classifies the emotions based on: angry ,sad,happy, fear ,disgust, surprise , neutral
> We will then use tensorflow + keras


* 2. Requirements: 
> python version: 3.8 - 3.11 
> libraries like : tensorflow ,opencv - python ,numpy , scikit-learn, matplotlib 

* 3. datset: 
> The dataset which I have used is : FER 2013 dataset --> It is the most common dataset for emotion detection 
> you can download it from kaggle --> FER2013
> It contains 2 parts i.e Train and test 
> few images are trained whereas few images which are not trained are tested. 

* 4. Algorithms Used (DETAILED)
>> 1. Convolutional Neural Networks (CNN)
CNN learns:
Edges
Shapes
Facial expressions
Core operations:
Convolution
Pooling
Feature extraction

>> 2. Transfer Learning
Instead of training from scratch:
Use pretrained weights
Add custom layers
Fine-tune for emotions
Huge advantages:
Less data required
Faster training
Higher accuracy

>> 3. Softmax Classifier
Converts raw scores to probabilities
Ensures sum = 1

>> 4. Backpropagation
Computes loss
Updates trainable weights
Uses gradient descent

>> 5. Dropout Regularization
Randomly disables neurons
Prevents memorization
Improves generalization