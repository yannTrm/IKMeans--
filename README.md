# IKMeansMinusPlus: Incremental K-Means Minus Plus

## Overview

IKMeansMinusPlus is an algorithm for clustering data, extending the conventional K-Means approach to iteratively refine clustering solutions. This repository contains an implementation of the IKMeansMinusPlus algorithm in Python, providing a flexible and efficient tool for clustering tasks.

## Usage

To use IKMeansMinusPlus for clustering, follow these steps:

1. **Instantiate the IKMeansMinusPlus class**: Initialize the clustering algorithm by specifying parameters such as the number of clusters, maximum iterations, and convergence tolerance.

2. **Fit the model**: Call the `fit` method with your dataset to compute the clustering using the IKMeansMinusPlus algorithm.

3. **Predict cluster labels**: Use the `predict` method to assign cluster labels to new data points based on the learned clustering model.

Alternatively, you can use the `fit_predict` method to perform both fitting and prediction in a single call.

## Example

```python
from IKMeansMinusPlus import IKMeansMinusPlus
import numpy as np

# Create an instance of IKMeansMinusPlus
ikmeans = IKMeansMinusPlus(n_clusters=3)

# Generate some sample data
X = np.random.rand(100, 2)

# Fit the model
ikmeans.fit(X)

# Predict cluster labels for new data
new_data = np.random.rand(10, 2)
labels = ikmeans.predict(new_data)
```

## Reference

This implementation of IKMeansMinusPlus is based on the paper:

Hassan-Ismkhan. "I-k-means−+: An Iterative Clustering Algorithm Based on an Enhanced Version of the k-means." 

## Note

Feel free to experiment with different parameter settings and datasets to explore the capabilities of IKMeansMinusPlus for your clustering tasks.
