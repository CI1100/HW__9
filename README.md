# HW_9
 Homework 9 Iris dataset

Pseudocode for Kmeans:

```
    y = randomly assigned classes for each datapoint X
	
	for i in range(steps):
		means = find_means(y)
		distances = find_distances(X, means)
		predictions = find_predictions(distances)
		if predictions == y:
		   break
		else 
			y = predictions
```

Pseudocode for ransack:


```  
    
	for n in range(number_of_steps):
	    maybeInliers, notInliers = split_dataset_randomly(X)
		means = train_kmeans(maybeInliers)
		distances = find_distanes(notInliers, means)
		addInliers = selelect_notInliers_where(distance < t)
		maybeInliers = maybeInliers + addInliers
		
		means = train_kmeans(maybeInliers)
		error = find_distances(maybeInliers, means)
		if error < best_error:
		   best_model = means
```