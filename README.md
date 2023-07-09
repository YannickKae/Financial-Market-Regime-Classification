# Financial Market Regime Classification

<p align="center">
  <img src="Value vs. Momentum.png" alt="Value vs. Momentum" style="width:100%">
</p>
<p align="center">
  <i>Cross-Asset Value vs. Momentum</i>
</p>

Following the financial crisis, quantitative value strategies have experienced a decade of relative underperformance, making it challenging to maintain commitment to them. This repository aimed to identify correponding regimes and implement tactical allocation changes based on four supervised methods for binary (Value vs. Momentum) regime classification.
Due to the results of [Fernández-Delgado et al. (2014)](https://jmlr.org/papers/v15/delgado14a.html), we focused on the following methods:

- Logistic Regression (base line model)
- Random Forest
- Support Vector Machine
- Multi-layer Perceptron

## Theory

### Value vs. Momentum

- **Value** is a strategy that focuses on securities with cheap fundamentals, such as low price-to-earnings ratios. The idea is that these undervalued securities will mean-revert in price. This approach is based on the idea that while the market may not accurately price a security's true value in the short term, over the long term its price will converge back to its true value.

- **Momentum** focuses on the recent price performance of securities, assuming that those with strong recent performance will continue to perform strong in the future, and vice versa. This approach is based on the idea that market trends and investor sentiment can drive the price of a security, and these trends may persist for some time before reversing. Side remark: Momentum is not Growth, the actual antagonist of Value.

Both strategies typically exhibit negative correlation, as value is countercyclical while momentum is procyclical. Consequently, we will divide the current market state along this cycle-based value vs. momentum dimension. That both are, on average, profitable — which may appear counterintuitive — can be explained by value operating on longer-term time frames, while momentum operates on shorter-term time frames.

### Methods

- **Logistic Regression** (LPB): This is one of the most fundamental models, especially for binary classification problems, such as the one in this case. The basic principle is to use a sigmoid function to model the probability of occurrence of a particular class y, given X. The sigmoid function takes real numbers as input and outputs values in the range [0, 1], allowing the output to be interpreted as a probability.

- **Support Vector Machines** (SVM): These are a family of supervised learning algorithms used for classification and regression problems. In the context of binary classification, SVMs attempt to find an optimal hyperplane that separates the two classes. Specifically, we use the C-Support Vector Classification (C-SVC) variant because it is particularly good at avoiding overfitting, similar to the hyperparameter 'C' in logistic regression. Technically, it controls the trade-off between achieving the largest possible margin and minimizing classification errors.

- **Random Forest** (RF): This is a bagging based ensemble technique, which creates an ensemble of multiple decision tree models and uses the majority decision of these trees for prediction. Each individual tree in the Random Forest is trained on a random subset of the training data (called bootstrap samples) and uses a random selection of features to find the best split at each node of the tree. This randomness leads to increased diversity among individual trees and helps decreasing variance.

- **Multi-layer Perceptron** (MLP): These are a type of artificial neural network consisting of at least three layers of neurons: an input layer, one or more "hidden" layers, and an output layer. Each layer is fully connected to the next, with each node receiving a weighted sum of inputs from the previous layer to which an activation function is applied.

## Data

From the [AQR Capital Management database](https://www.aqr.com/Insights/Datasets/Century-of-Factor-Premia-Monthly), we had access to monthly returns for both strategies dating back to 1926. Using this data, we extracted additional features, including rolling cumulative performance, volatility, and correlation. We supplemented them with key financial macroeconomic indicators sourced from the [FRED database](https://stlouisfed.shinyapps.io/macro-snapshot/#financial).

The data from the latter varied in length, confronting us with a challenging trade-off - should we prioritize having more features or more historical data? To resolve this, we conducted a bootstrapping test. The purpose of this was to ascertain whether shorter feature time series were redundant (null hypothesis) and therefore could be discarded in favor of having more historical data, or not (alternative hypothesis).
The null hat to be rejected multiple times. Consequently, we repeated the models' training and testing process multiple times, using different data sets along the described trade-off.

This iterative approach gave us the opportunity to observe and understand how this trade-off played out in our specific case.

## Results

Since our data was very unbalanced and we could not always assume with sufficient probability that a shorter feature time series is redundant, we ran the training & testing procedure multiple times on different data.
This allowed us to see how the trade-off between more features & less data vs. less features & more data behaves.

### Time Period 1 (2003 - 2023)

### Time Period 2 (1976 - 2023)

### Time Period 3 (1971 - 2023)

### Time Period 4 (1965 - 2023)

### Time Period 5 (1954 - 2023)

### Time Period 6 (1929 - 2023)

#### Links

- [Federal Reserve Economic Data](https://fred.stlouisfed.org/)
- [Online Data by Robert Shiller](http://www.econ.yale.edu/~shiller/data.htm)
- [How Do Factor Premia Vary Over Time? (Factor Data Monthly)](https://www.aqr.com/Insights/Datasets/Century-of-Factor-Premia-Monthly)
- [Classifying market regimes (Macrosynergy Research)](https://research.macrosynergy.com/classifying-market-regimes/)
- [Time Series Forecasting as Supervised Learning](https://machinelearningmastery.com/time-series-forecasting-supervised-learning/)
- [Random Forest for Time Series Forecasting](https://machinelearningmastery.com/random-forest-for-time-series-forecasting/)
- [How to Develop Multilayer Perceptron Models for Time Series Forecasting](https://machinelearningmastery.com/how-to-develop-multilayer-perceptron-models-for-time-series-forecasting/)
- [Return Data](https://www.aqr.com/Insights/Datasets/Century-of-Factor-Premia-Monthly)
