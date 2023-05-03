# Market Regime Classification

<p align="center">
  <img src="Header.jpeg" alt="Chameleons adapt to their environment" style="width:100%">
</p>
<p align="center">
  <i>Chameleons adapt to their environment</i>
</p>

Following the financial crisis, quantitative value strategies have experienced a decade of underperformance, making it challenging to maintain commitment to them. This repository aims to identify correponding regimes and implement tactical allocation changes by evaluating the performance of three supervised methods for binary regime classification. Due to the results of Fernández-Delgado et al. (2014), we focus in the following methods:

- Linear Probability Model (base line model)
- Random Forest
- Support Vector Machine
- Multi-layer Perceptron

## Theory

### Value vs. Momentum

**Value** investing is a strategy focusing on securities with cheap fundamental indicators, such as low price-to-earnings or price-to-book ratios. The belief is that these undervalued securities will appreciate in price eventually. This approach relies on the idea that the market may not accurately price a security's true value in the short term, but its price will converge to its true value in the long term.

**Momentum** investing focuses on the recent price performance of securities, assuming that those with strong recent performance will continue to perform strong in the future, and vice versa. This approach is based on the idea that market trends and investor sentiment can drive the price of a security, and these trends may persist for some time before reversing.

Both strategies typically exhibit negative correlation, as value is countercyclical while momentum is procyclical. Consequently, we will divide the current market state along this cycle-based value vs. momentum dimension. That both are, on average, profitable — which may appear counterintuitive — can be explained by value operating on longer-term time frames, while momentum operates on shorter-term time frames.

### Methods

**Linear Probability Model** (LPB): This is a simple linear regression model applied to binary classification problems. It models the probability of a binary outcome as a linear function of the input features. Although it may not be the most accurate method for classification tasks, it serves as a useful baseline model to compare against more sophisticated techniques.

**Random Forest** (RF): This is an ensemble learning method that constructs multiple decision trees and combines their predictions to produce a more accurate and robust result. By averaging the predictions of the individual trees, it can reduce overfitting and improve generalization. Random Forests are versatile and can be used for both classification and regression tasks, making them a popular choice for various machine learning problems.

**Support Vector Machines** (SVM): These are a class of supervised learning algorithms used for classification and regression tasks. In classification problems, SVM aims to find the best hyperplane that separates the different classes with the maximum margin. It can efficiently handle high-dimensional data and is particularly effective when the number of features is greater than the number of samples. SVMs can be used with various kernel functions, which allows for capturing complex, nonlinear relationships between the input features and the target variable.

**Multi-layer Perceptron** (MLP): These are a type of feedforward artificial neural network (NN) that consists of multiple layers of nodes or neurons. MLPs include an input layer, one or more hidden layers, and an output layer. They are trained using a backpropagation algorithm, which minimizes the error between the predicted output and the actual target by adjusting the weights of the connections between the neurons. MLPs can be used for a wide range of tasks, including classification and regression problems, and they are capable of capturing complex, nonlinear relationships between input features and target variables.

#### Useful Links

- [Federal Reserve Economic Data](https://fred.stlouisfed.org/)
- [Online Data by Robert Shiller](http://www.econ.yale.edu/~shiller/data.htm)
- [How Do Factor Premia Vary Over Time? (Factor Data Monthly)](https://www.aqr.com/Insights/Datasets/Century-of-Factor-Premia-Monthly)
- [Classifying market regimes (Macrosynergy Research)](https://research.macrosynergy.com/classifying-market-regimes/)
- [Time Series Forecasting as Supervised Learning](https://machinelearningmastery.com/time-series-forecasting-supervised-learning/)
- [Random Forest for Time Series Forecasting](https://machinelearningmastery.com/random-forest-for-time-series-forecasting/)
- [How to Develop Multilayer Perceptron Models for Time Series Forecasting](https://machinelearningmastery.com/how-to-develop-multilayer-perceptron-models-for-time-series-forecasting/)
