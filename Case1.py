### CASE 1 ###

import numpy as np
import pandas as pd
from scipy import stats

# Ratings
ratings = ['AAA', 'AA', 'A', 'BBB', 'BB', 'B', 'CCC', 'D']
rating_to_idx = {r: i for i, r in enumerate(ratings)}

# Transition matrix (probabilities)
transition_matrix = np.array([
    [0.91115, 0.08179, 0.00607, 0.00072, 0.00024, 0.00003, 0.00000, 0.00000],  # AAA
    [0.00844, 0.89626, 0.08954, 0.00437, 0.00064, 0.00036, 0.00018, 0.00021],  # AA
    [0.00055, 0.02595, 0.91138, 0.05509, 0.00499, 0.00107, 0.00045, 0.00052],  # A
    [0.00031, 0.00147, 0.04289, 0.90584, 0.03898, 0.00708, 0.00175, 0.00168],  # BBB
    [0.00007, 0.00044, 0.00446, 0.06741, 0.83274, 0.07667, 0.00895, 0.00926],  # BB
    [0.00008, 0.00031, 0.00150, 0.00490, 0.05373, 0.82531, 0.07894, 0.03523],  # B
    [0.00000, 0.00015, 0.00023, 0.00091, 0.00388, 0.07630, 0.83035, 0.08818],  # CCC
])

# Forward bond values at t=1
forward_values = np.array([99.50, 98.51, 97.53, 92.77, 90.48, 88.25, 77.88, 60.00])

# Portfolio allocations (in EUR millions)
portfolio_I = {'AAA': 900, 'AA': 450, 'BBB': 150}
portfolio_II = {'BB': 900, 'B': 525, 'CCC': 75}

# Correlation values
rho_values = [0.0, 0.33, 0.66, 1.0]

##### Q1

print("##### Q1 #####")

### 1. Default Threshold Check

# Default Threshold Check for BBB rating

# BBB to Default probability from transition matrix
bbb_idx = rating_to_idx['BBB']
default_idx = rating_to_idx['D']
prob_bbb_default = transition_matrix[bbb_idx, default_idx]

print(f"BBB → Default probability: {prob_bbb_default:.5f} = {prob_bbb_default*100:.3f}%")

# Calculate Z-score threshold
# P(X_i < Z_threshold) = prob_default
# For standard normal: Z_threshold = Φ^(-1)(prob_default)
z_threshold_bbb_default = stats.norm.ppf(prob_bbb_default)

print(f"\nBBB → Default Z-threshold: {z_threshold_bbb_default:.6f}")
print(f"Verification: Φ({z_threshold_bbb_default:.6f}) = {stats.norm.cdf(z_threshold_bbb_default):.5f}")

# Calculate cumulative transition probabilities and thresholds for all ratings

def calculate_thresholds(transition_matrix):
    """Calculate Z-score thresholds for all rating transitions"""
    n_from_ratings = transition_matrix.shape[0]  # 7 (excluding D as source)
    n_to_ratings = transition_matrix.shape[1]    # 8 (including D)
    
    thresholds = np.zeros((n_from_ratings, n_to_ratings))
    
    for i in range(n_from_ratings):
        # Cumulative probabilities
        cum_probs = np.cumsum(transition_matrix[i, :])
        
        # Thresholds: inverse CDF of cumulative probabilities
        for j in range(n_to_ratings):
            if cum_probs[j] >= 1.0:
                thresholds[i, j] = np.inf
            else:
                thresholds[i, j] = stats.norm.ppf(cum_probs[j])
    
    return thresholds

thresholds = calculate_thresholds(transition_matrix)

# Display and verify
print("\n" + "="*70)
print("CORRECTED RATING TRANSITION THRESHOLDS")
print("="*70)
threshold_df = pd.DataFrame(
    thresholds,
    index=ratings[:-1],
    columns=ratings
)
print(threshold_df.round(4))

print(f"\nBBB → Default threshold: {thresholds[bbb_idx, default_idx]:.6f}")



### 2. Convergence Check: Portfolio II, rho=33%

def simulate_portfolio_loss(portfolio_alloc, rho, n_simulations, seed=None):
    """Simulate portfolio value for single issuer per rating class"""
    if seed is not None:
        np.random.seed(seed)
    
    portfolio_values = np.zeros(n_simulations)
    
    for sim in range(n_simulations):
        Y = np.random.standard_normal()
        total_value = 0
        
        for rating, amount in portfolio_alloc.items():
            rating_idx = rating_to_idx[rating]
            epsilon = np.random.standard_normal()
            
            # Asset return
            X_i = np.sqrt(rho) * Y + np.sqrt(1 - rho) * epsilon
            
            # Determine new rating by comparing X_i to thresholds
            new_rating_idx = 0
            for j in range(len(thresholds[rating_idx])):
                if X_i <= thresholds[rating_idx, j]:
                    new_rating_idx = j
                    break
            
            # Forward value for new rating
            new_value = forward_values[new_rating_idx]
            
            # Number of bonds
            current_price = forward_values[rating_idx]
            n_bonds = amount / current_price
            
            total_value += n_bonds * new_value
        
        portfolio_values[sim] = total_value
    
    return portfolio_values

# Helper function to calculate VaR
def calculate_var(portfolio_values, confidence_level):
    """Calculate Value at Risk (loss from initial value)"""
    initial_value = 1500  # EUR millions
    losses = initial_value - portfolio_values
    var = np.percentile(losses, confidence_level * 100)
    return var

# Run convergence check: 3 simulations with different seeds
print("="*70)
print("CONVERGENCE CHECK: Portfolio II, ρ=33%")
print("="*70)

test_seeds = [42, 123, 999]
n_sim_test = 10000  # Start with 10k simulations

print(f"\nTesting with N = {n_sim_test:,} simulations")
print("\nSeed | 99.5% VaR (EUR M)")
print("-" * 30)

var_995_results = []

for seed in test_seeds:
    pf_values = simulate_portfolio_loss(portfolio_II, rho=0.33, n_simulations=n_sim_test, seed=seed)
    var_995 = calculate_var(pf_values, 0.995)
    var_995_results.append(var_995)
    print(f"{seed:4d} | {var_995:8.2f}")

# Calculate range and relative variation
var_range = max(var_995_results) - min(var_995_results)
var_mean = np.mean(var_995_results)
relative_variation = (var_range / var_mean) * 100

print("\n" + "-" * 30)
print(f"Range: {var_range:.2f} EUR M")
print(f"Mean: {var_mean:.2f} EUR M")
print(f"Relative variation: {relative_variation:.2f}%")