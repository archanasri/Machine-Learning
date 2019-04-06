# Perceptron

The python script ml.py implements different variants of perceptron such as:
1. Simple Perceptron,
2. Perceptron with dynamic learning rate,
3. Margin Perceptron, 
4. Averaged Perceptron and
5. Aggresive Perceptron with dynamic learning rate.

All these variants of perceptron are implemented from scratch in python.


The command to run the script is

python ml.py phishing_train.txt phishing_dev.txt training00.txt training01.txt training02.txt training03.txt training04.txt phishing_test.txt

phishing_train.txt is the file for training the perceptrons and phishing_test.txt is the text file for testing the perceptrons.
Text files training00.txt training01.txt training02.txt training03.txt training04.txt are used for 5 fold cross validation.
