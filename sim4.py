import pandas as pd
import numpy as np
from scipy.stats import binned_statistic
from sklearn.preprocessing import KBinsDiscretizer

def mse_quantization(data, n_buckets):
    discretizer = KBinsDiscretizer(n_bins=n_buckets, encode='ordinal', strategy='quantile')
    discretizer.fit(data.reshape(-1, 1))
    boundaries = discretizer.bin_edges_[0]
    return boundaries[1:-1]  # Exclude the first and last boundaries

def log_likelihood_quantization(fico_scores, defaults, n_buckets):
    def calculate_ll(boundaries):
        bins = np.digitize(fico_scores, boundaries)
        ll = 0
        for i in range(n_buckets):
            mask = bins == i
            ni = np.sum(mask)
            if ni > 0:
                ki = np.sum(defaults[mask])
                pi = ki / ni
                if 0 < pi < 1:
                    ll += ki * np.log(pi) + (ni - ki) * np.log(1 - pi)
        return ll

    def optimize_boundary(start, end, remaining_buckets):
        if remaining_buckets == 1:
            return [end], calculate_ll([start, end])

        best_boundaries = None
        best_ll = float('-inf')

        for split in range(start + 1, end):
            left_boundaries, left_ll = optimize_boundary(start, split, 1)
            right_boundaries, right_ll = optimize_boundary(split, end, remaining_buckets - 1)
            
            current_boundaries = left_boundaries + right_boundaries
            current_ll = left_ll + right_ll

            if current_ll > best_ll:
                best_ll = current_ll
                best_boundaries = current_boundaries

        return best_boundaries, best_ll

    min_score, max_score = int(fico_scores.min()), int(fico_scores.max())
    optimal_boundaries, _ = optimize_boundary(min_score, max_score, n_buckets)
    return optimal_boundaries[:-1]  # Exclude the last boundary

# Load the data
df = pd.read_csv('Task 3 and 4_Loan_Data.csv')
fico_scores = df['fico_score'].values
defaults = df['default'].values

# Number of buckets
n_buckets = 10

# MSE Quantization
mse_boundaries = mse_quantization(fico_scores, n_buckets)

# Log-Likelihood Quantization
ll_boundaries = log_likelihood_quantization(fico_scores, defaults, n_buckets)

# Create rating maps
def create_rating_map(boundaries):
    boundaries = [300] + list(boundaries) + [850]
    rating_map = {i: f"Rating {n_buckets - i}" for i in range(n_buckets)}
    return lambda x: rating_map[np.digitize(x, boundaries) - 1]

mse_rating_map = create_rating_map(mse_boundaries)
ll_rating_map = create_rating_map(ll_boundaries)

# Apply rating maps to the data
df['MSE_Rating'] = df['fico_score'].apply(mse_rating_map)
df['LL_Rating'] = df['fico_score'].apply(ll_rating_map)

# Print results
print("MSE Bucket Boundaries:", mse_boundaries)
print("Log-Likelihood Bucket Boundaries:", ll_boundaries)

print("\nSample of ratings:")
print(df[['fico_score', 'MSE_Rating', 'LL_Rating', 'default']].sample(10))

# Calculate default rates for each rating
def calculate_default_rates(df, rating_column):
    return df.groupby(rating_column)['default'].agg(['mean', 'count']).sort_index(ascending=False)

print("\nMSE Rating Default Rates:")
print(calculate_default_rates(df, 'MSE_Rating'))

print("\nLog-Likelihood Rating Default Rates:")
print(calculate_default_rates(df, 'LL_Rating'))