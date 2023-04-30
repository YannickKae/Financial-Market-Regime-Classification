# Market Regime Classification

Following the financial crisis, quantitative value strategies have experienced a decade of underperformance, making it challenging to maintain commitment to them. This repository aims to identify correponding regimes and implement tactical allocation changes by evaluating the performance of various machine learning models for regime classification:

- Random Forests
- Wasserstein k-means Clustering
- Hidden Markov Models
- Gaussian Mixture Models

## Value vs. Momentum

Value investing is a strategy focusing on securities with cheap fundamental indicators, such as low price-to-earnings or price-to-book ratios. The belief is that these undervalued securities will appreciate in price eventually. This approach relies on the idea that the market may not accurately price a security's true value in the short term, but its price will converge to its true value in the long term.

Momentum investing focuses on the recent price performance of securities, assuming that those with strong recent performance will continue to perform strong in the future, and vice versa. This approach is based on the idea that market trends and investor sentiment can drive the price of a security, and these trends may persist for some time before reversing.

Value and momentum strategies typically exhibit negative correlation, as value is countercyclical while momentum is procyclical. Consequently, our focus will be on these two strategies. That both strategies are profitable on average can be attributed to value operating on longer-term time frames, while momentum operates on shorter-term time frames.

## Methods

**Random Forests** is a machine learning algorithm that constructs numerous decision trees during training and outputs the majority vote of individual trees for classification or the mean prediction for regression. This method is known for its robustness and ability to handle complex data structures.

**Wasserstein k-means Clustering** is an unsupervised learning method that groups data points based on similarity, minimizing the Wasserstein distance between clusters. This method is particularly useful when dealing with data distributions that are not easily separated by traditional distance metrics.

**Hidden Markov Models (HMMs)** are statistical models representing a system that transitions between hidden states over time, with each state emitting observable data. HMMs are particularly useful in modeling time series data, as they capture the underlying structure and dynamics of the data, making them well-suited for financial time series analysis.

**Gaussian Mixture Models (GMMs)** are probabilistic models that represent data as a mixture of multiple Gaussian distributions. This method estimates the parameters of these Gaussian distributions and the probabilities of each data point belonging to each distribution. GMMs can be used for clustering, classification, or density estimation and are especially useful when dealing with data that has multiple underlying distributions.

### Useful Links

- [Federal Reserve Economic Data](https://fred.stlouisfed.org/)
- [Online Data by Robert Shiller](http://www.econ.yale.edu/~shiller/data.htm)
- [How Do Factor Premia Vary Over Time? (Factor Data Monthly)](https://www.aqr.com/Insights/Datasets/Century-of-Factor-Premia-Monthly)
- [Classifying market regimes (Macrosynergy Research)](https://research.macrosynergy.com/classifying-market-regimes/)
