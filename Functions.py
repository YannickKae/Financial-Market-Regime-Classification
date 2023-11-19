# The following function can be used to test whether the Sharpe Ratio of your trading model (sample2) is statistically significantly higher than that of the benchmark (sample1).
# Pass returns to the function

def sharpe_ratio_bootstrap_test(sample1, sample2, n_permutations=1000000):
    
    # Bootstrap sampling from sample1 and compute Sharpe ratios
    sharpe_ratios = []
    n = len(sample1)
    for _ in range(n_permutations):
        bootstrap_sample = np.random.choice(sample1, size=n, replace=True)
        mean_return = np.mean(bootstrap_sample)
        std_return = np.std(bootstrap_sample)
        sharpe_ratio = mean_return / std_return
        sharpe_ratios.append(sharpe_ratio)

    # Compute the observed Sharpe ratio for sample2
    observed_mean_return = np.mean(sample2)
    observed_std_return = np.std(sample2)
    observed_sharpe_ratio = observed_mean_return / observed_std_return

    # Calculate p-value: proportion of null Sharpe ratios more extreme than observed Sharpe ratio
    p_value = (np.abs(sharpe_ratios) >= np.abs(observed_sharpe_ratio)).mean()
    
    # Output
    print(f"p-value: {p_value*100:.2f}%")  # Print the calculated p-value as a percentage with two decimal points

    # Plot the null distribution of Sharpe ratios
    plt.figure(figsize=(11, 6), dpi=100)
    plt.hist(sharpe_ratios, bins=100, color='black')
    plt.axvline(observed_sharpe_ratio, color='red', linestyle='dashed', linewidth=2,
                label=f'Model Sharpe Ratio: {observed_sharpe_ratio:.2f}')
    plt.legend()
    plt.xlabel('Sharpe Ratio')
    plt.ylabel('frequency')
    plt.title('Null Distribution of Benchmark Sharpe Ratios')
    plt.show()


# The function below can be used to test, whether differences between two time series are du to noise or not.
# It takes two time series, scales them uniformly to ensure scale & correlation direction invariance, extracts the difference time series, and then circularly bootstraps it 100,000 times.
# This ensures that any autodependencies are preserved in this time series context. The optimal block length is chosen according to Patton et al. (2009).
# Then, these difference time series are added to the first time series and the corresponding R^2 is computed with the first time series ifself.
# This generates a null distribution under the assumption that the true R^2 is 1.

def standardize_series(s):
    scaler = StandardScaler()
    s_values_scaled = scaler.fit_transform(s.values.reshape(-1, 1))
    return pd.Series(s_values_scaled.flatten(), index=s.index)

def r_squared_bootstrap_test(ts1, ts2, n_permutations):
    
    # Standardize the time series
    ts1 = standardize_series(ts1)
    ts2 = standardize_series(ts2)
    
    if np.corrcoef(ts1, ts2)[0, 1] < 0:
        ts2 = ts2 * -1

    # Calculate observed R^2
    observed_r_squared = np.square(np.corrcoef(ts1, ts2)[0, 1])
    
    # Calculate observed R^2
    observed_r_squared = np.square(np.corrcoef(ts1, ts2)[0, 1])

    # Calculate differences between ts2 and ts1
    differences = ts2 - ts1
    differences_values = differences.values

    # Calculate optimal block length
    opt_block_length_df = optimal_block_length(differences_values)
    opt_block_length = int(opt_block_length_df['circular'])  # Use the optimal block length
    
    # Check if the optimal block length is zero, if so, set it to 1
    if opt_block_length == 0:
        opt_block_length = 1

    # Initialize circular block bootstrap with optimal block length
    bs = CircularBlockBootstrap(opt_block_length, differences_values)

    # Compute R^2 for bootstrapped series and generate null distribution
    null_r_squared = []
    for _, bs_diffs in zip(range(n_permutations), bs.bootstrap(n_permutations)):
        # Generate new ts2 by adding bootstrapped differences to ts1
        # Ensure that ts2_new has the same length as ts1 by discarding extra values or padding with zeros
        bs_diffs_trimmed = bs_diffs[0][0][:len(ts1)]
        ts2_new = ts1 + pd.Series(bs_diffs_trimmed, index=ts1.index)
        r_squared = np.square(np.corrcoef(ts1, ts2_new)[0, 1])
        null_r_squared.append(r_squared)  # Store R^2 for each bootstrapped series

    # Compute p-value: proportion of null R^2 values greater than or equal to observed R^2
    p_value = np.mean(np.array(null_r_squared) <= observed_r_squared)

    # Output
    print(f"p-value: {p_value*100:.2f}%")  # Print the calculated p-value as a percentage with two decimal points
    
    # Set figure size and dpi
    plt.figure(figsize=(11, 6), dpi=100)

    # Plot null distribution
    plt.hist(null_r_squared, bins=100, color='black')  # Plot the histogram of null R^2 values
    plt.axvline(observed_r_squared, color='red', linestyle='dashed', linewidth=2, label=f'Observed R^2: {observed_r_squared:.2f}')
    plt.legend()
    plt.xlabel('R^2')
    plt.ylabel('Frequency')
    plt.title('Null distribution of R^2')
    plt.show()


# A more general test function, which allows to choose from pearson's correlation, kendall's tau or spearman's rho

def redundancy_bootstrap_test(ts1, ts2, n_permutations = 100000, test_statistic = 'correlation'):
    
    def standardize_series(s):
        scaler = StandardScaler()
        s_values_scaled = scaler.fit_transform(s.values.reshape(-1, 1))
        return pd.Series(s_values_scaled.flatten(), index=s.index)
    
    # Standardize the time series
    ts1 = standardize_series(ts1)
    ts2 = standardize_series(ts2)
    
    # Select the test statistic function
    if test_statistic == 'correlation':
        stat_func = lambda x, y: np.corrcoef(x, y)[0, 1]
    elif test_statistic == 'kendall':
        stat_func = lambda x, y: kendalltau(x, y)[0]
    elif test_statistic == 'spearman':
        stat_func = lambda x, y: spearmanr(x, y)[0]
    else:
        raise ValueError("Invalid test statistic. Choose 'correlation', 'kendall', or 'spearman'.")
    
    # Adjust ts2 based on the chosen statistic
    if stat_func(ts1, ts2) < 0:
        ts2 = ts2 * -1

    # Calculate observed R^2
    observed_stat = stat_func(ts1, ts2)

    # Calculate differences between ts2 and ts1
    differences = ts2 - ts1
    differences_values = differences.values
    
    # Check if differences are all zeros
    if np.all(differences_values == 0):
        #print("Both time series are redundant (identical after standardization).")
        return 1

    # Calculate optimal block length
    opt_block_length_df = optimal_block_length(differences_values)
    opt_block_length = int(opt_block_length_df['circular'])  # Use the optimal block length
    
    # Check if the optimal block length is zero, if so, set it to 1
    if opt_block_length == 0:
        opt_block_length = 1

    # Initialize circular block bootstrap with optimal block length
    bs = CircularBlockBootstrap(opt_block_length, differences_values)

    # Compute R^2 for bootstrapped series and generate null distribution
    null_stats = []
    for _, bs_diffs in zip(range(n_permutations), bs.bootstrap(n_permutations)):
        # Generate new ts2 by adding bootstrapped differences to ts1
        # Ensure that ts2_new has the same length as ts1 by discarding extra values or padding with zeros
        bs_diffs_trimmed = bs_diffs[0][0][:len(ts1)]
        ts2_new = ts1 + pd.Series(bs_diffs_trimmed, index=ts1.index)
        stat = stat_func(ts1, ts2_new)
        null_stats.append(stat)  # Store Stat for each bootstrapped series

    # Compute p-value: proportion of null R^2 values greater than or equal to observed R^2
    p_value = np.mean(np.array(null_stats) <= observed_stat)
    
    return p_value

    # Output
    print(f"Observed {test_statistic}: {observed_stat}, p-value: {p_value}")

    # Plotting
    plt.figure(figsize=(11, 6), dpi=100)
    plt.hist(null_stats, bins=100, color='black')
    plt.axvline(observed_stat, color='red', linestyle='dashed', linewidth=2, label=f'Observed {test_statistic}: {observed_stat:.2f}')
    plt.legend()
    plt.xlabel(test_statistic)
    plt.ylabel('Frequency')
    plt.title(f'Null distribution of {test_statistic}')
    plt.show()


# The following function takes a set of variables, computes the pairwise R^2 of one of them to all the others and ranks them.

def plot_r_squared(dataframe, column_name):
    # Calculate the correlation matrix
    corr_matrix = dataframe.corr()

    # Select the correlations related to the specified column
    correlations = corr_matrix[column_name]

    # Compute R^2 from correlations
    r_squared = correlations.apply(lambda x: x ** 2)

    # Sort the R^2 values
    sorted_r_squared = r_squared.sort_values(ascending=True)

    # Plot the R^2 values
    plt.figure(figsize=(11, 11), dpi = 100)
    sorted_r_squared.drop(column_name).plot(kind='barh', color='black')  # Exclude the specified column as its R^2 with itself is 1
    plt.xlabel('R^2')
    plt.title(f'Pairwise R^2 with {column_name}')
    plt.show()
