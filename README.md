# Title 
Detecting Betrayal by analyzing the features present in the communication

---
## Abstract

The goal of the research paper is to predict whether a Diplomacy game would end in betrayal by extracting features from the messages exchanged between players during the game. For obtaining the predictions authors of the original paper chose a logistic regression model which they trained using 5 fold cross validation. Our idea for this project is to apply different methods from the ones mentioned in the research paper in order to try to obtain better accuracy results. We think that by using a different approach such as implementing a Neural Network, we can create a better model that will answer the 2 research questions we have in mind. We are going to be using the same dataset as in the research paper in order to be able to compare the results accurately.

---
## Research Questions

1. *Using linguistic cues, is it possible to detect betrayal?*
2. *What is the probability of betrayal happening next season given the messages received so far?*
---
## Proposed dataset
**Diplomacy Betrayal Dataset**: This dataset is the same dataset used in the paper given to us. Due to the nature of our paper, there are no other datasets that have similar properties(that we can find). We are not going to modify or augment the data as doing so might disrupt the relationships between the messages in an unpredictable way.

---
## Methods
We plan to implement two different Neural Network architectures and compare their results with the regression method used in the paper. These methods are:
 - **Fully Connected Linear Neural Network:** We plan to use a basic linear NN architecture consisting of 3 layers of neurons. Input to the network will be a vector of the average message features for a given season. The last layer will have a single neuron which will output the probability that a betrayal will occur next season. Exact parameters of the model such as the number of neurons in the hidden layer, activation function of the neurons and hyperparameters like the learning rate will be picked according to the model’s cross-validation accuracy.
 - **Recurrent Neural Network:** In this architecture, we will implement a RNN which will again take the message features as input and will output a probability of betrayal similar to our linear model. Main difference between this model and our linear model is that the Linear model only takes a single season as an input which means that it has no memory of previous seasons. By using an RNN, we are making sure that the model also remembers the previous seasons and takes them into account while it generates an output. We are hoping that this temporal memory will help the model predict an outcome.
 
---
## Proposed timeline

---

## Organization within the team
1. Creating a code skeleton for model training. (*Marija and Görkem*)
Loading data, Splitting data, Empty model creation…
2. Linguistic feature selection to be used as an input vector.(*Marija and Mert*)
3. Code RNN and Linear model and test the accuracy for research question 1.(*Mert and Görkem*)
4. Visualize the results and compare them with the Linguistic Harbingers of Betrayal: A Case Study on an Online Strategy Game.(*Görkem and Marija*)
5. Repeat  3 and 4 for research question 2.(*Marija and Mert*)
6. Repeat 2-5 until we have a plausible conclusion.(*Split will be determined after we have a result*)
7. Write a report about our results.(*As a team*)
8. Film a video. (*As a team*)
