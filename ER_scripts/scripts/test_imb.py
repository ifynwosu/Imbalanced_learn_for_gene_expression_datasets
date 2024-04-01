import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.datasets import make_classification

# Create a classification dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=2, n_redundant=0, n_clusters_per_class=2, random_state=42)

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Create a list of resampling strategies
resampling_strategies = ['cross_val_score', 'RepeatedStratifiedKFold']

# Create a dictionary to store the results
results = {}

# Iterate over the resampling strategies
for resampling_strategy in resampling_strategies:

    # Create a list to store the scores
    scores = []

    # Iterate over the models
    for model in models:

        # Fit the model to the training data
        model.fit(X_train, y_train)

        # Evaluate the model on the test data
        score = model.score(X_test, y_test)

        # Append the score to the list
        scores.append(score)

    # Store the scores in the dictionary
    results[resampling_strategy] = scores

# Print the results
for resampling_strategy, scores in results.items():
    print(f'{resampling_strategy}: {np.mean(scores)}')