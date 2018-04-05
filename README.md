# Perceptron

The python script ml.py implements different variants of perceptron such as:
Simple Perceptron,
Perceptron with dynamic learning rate,
Margin Perceptron, 
Averaged Perceptron and
Aggresive Perceptron with dynamic learning rate.


The command to run the script is

python ml.py phishing_train.txt phishing_dev.txt training00.txt training01.txt training02.txt training03.txt training04.txt phishing_test.txt

phishing_train.txt is the file for training the perceptrons and phishing_test.txt is the text file for testing the perceptrons.
Text files training00.txt training01.txt training02.txt training03.txt training04.txt are used for 5 fold cross validation.
