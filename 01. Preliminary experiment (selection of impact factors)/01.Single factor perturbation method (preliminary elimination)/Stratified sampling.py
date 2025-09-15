"""
This script performs stratified sampling within a given interval [lower, upper],
generating 'num_samples' uniformly distributed sample points.
Used for random sampling of disturbed variables under the single-factor perturbation method.
"""

import random

def stratified_sampling(lower, upper, num_samples):
    """
    Perform stratified sampling within the interval [lower, upper], 
    generating num_samples uniformly distributed sample points.
    
    Args:
        lower (float): Lower bound of the interval
        upper (float): Upper bound of the interval
        num_samples (int): Number of samples to generate
    
    Returns:
        list: A list containing the sampled results
    """
    range_width = upper - lower
    step = range_width / num_samples
    
    samples = []
    for i in range(num_samples):
        sub_lower = lower + i * step
        sub_upper = sub_lower + step
        samples.append(random.uniform(sub_lower, sub_upper))
    
    random.shuffle(samples)
    return samples

if __name__ == "__main__":
    # Parameter settings
    LOWER = 1        # Lower bound of the interval
    UPPER = 100      # Upper bound of the interval
    NUM_SAMPLES = 20 # Number of samples
    
    # Perform stratified sampling
    samples = stratified_sampling(LOWER, UPPER, NUM_SAMPLES)

print(samples)
