# Market Regime Classification
In the aftermath of the financial crisis, quantitative value strategies have experienced a decade of underperformance, what makes it unbearable to stick to it. Therefore, this repository aims to recognize such periods and make appropriate tactical allocation changes by comparing the performance of various machine learning models for regime classification:
- Random Forests
- Wasserstein k-means Clustering
- Hidden Markov Models
- Gaussian Mixture Models
## Value vs. Momentum
Value investing is a strategy that involves selecting securities with cheap fundamental indicators, such as low price-to-earnings or price-to-book ratios, with the belief that these undervalued securities will eventually appreciate in price. This approach is based on the idea that the market may not accurately price a security's true value in the short term, but over time, its price will converge back to its true value.
Momentum investing is a strategy that focuses on the recent price performance of securities, with the assumption that those that have performed well in the recent past will continue to do so in the future, and vice versa. This approach is based on the idea that market trends and investor sentiment can drive the price of a security, and that these trends can persist for some time before reversing.
Both strategies tend to be negatively correlated due to the fact that Value is anti-cyclical while Momentum is pro-cyclical. 
## Methods
Random Forests is a machine learning algorithm that constructs a multitude of decision trees during the training process and outputs the majority vote of the individual trees for classification or the mean prediction for regression. This method is known for its robustness and ability to handle complex data structures.
Wasserstein k-means Clustering is an unsupervised learning method that groups data points based on their similarity in a way that minimizes the Wasserstein distance between these clusters. This method is particularly useful for dealing with data distributions that are not easily separated by traditional distance metrics.
Hidden Markov Models (HMMs) are statistical models that represent a system that transitions between hidden states over time, with each state emitting observable data. HMMs are particularly useful in modeling time series data, as they capture the underlying structure and dynamics of the data, making them well-suited for financial time series analysis.
Gaussian Mixture Models (GMMs) are a probabilistic model that represents the data as a mixture of multiple Gaussian distributions. This method estimates the parameters of these Gaussian distributions and the probabilities of each data point belonging to each distribution. GMMs can be used for clustering, classification, or density estimation, and can be especially useful when dealing with data that has multiple underlying distributions.
