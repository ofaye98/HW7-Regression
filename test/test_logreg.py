"""
Write your unit tests here. Some tests to include are listed below.
This is not an exhaustive list.

- check that prediction is working correctly
- check that your loss function is being calculated correctly
- check that your gradient is being calculated correctly
- check that your weights update during training
"""

# Imports
import pytest
import numpy as np
from regression.logreg import LogisticRegressor

def test_prediction():
	model = LogisticRegressor(num_feats=2) # create instance of logistic regression model with 2 features
	model.W = np.array([1.0, -1.0, 0.5]) # set weights to known values for testing

	# create a small test dataset with 3 data points and 2 features
	X = np.array([
		[2.0, 1.0],
		[0.0, 0.0],
		[1.0, 3.0]
	])

	y_pred = model.make_prediction(X) # compute predicted probabilities for the test dataset

	z = X @ model.W # compute linear combination of inputs and weights (matrix multiplication)
	expected = 1 / (1 + np.exp(-z)) # compute expected probabilities using sigmoid function

	assert np.allclose(y_pred, expected), "Your predicted probabilities do not match the expected values"

def test_loss_function():
	model = LogisticRegressor(num_feats=2) 

	y_true = np.array([1, 0, 1, 0]) # true labels for 4 data points
	y_pred = np.array([0.9, 0.2, 0.8, 0.1]) # predicted probabilities for the 4 data points
	y_pred_clipped = np.clip(y_pred, 1e-15, 1 - 1e-15) # prevent predicted probabilities from being exactly 0 or 1, which would cause issues with the log function
	
	expected = -np.mean(
		y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped) # compute expected binary cross entropy loss for the 4 data points
	)
	loss = model.loss_function(y_true, y_pred) # compute loss using the model's loss function

	assert np.isclose(loss, expected), "Your loss function's result does not match the expected value"

def test_gradient():
	model = LogisticRegressor(num_feats=2)
	model.W = np.array([0.2, -0.3, 0.1])

	X = np.array([
		[1.0, 2.0, 1.0],
		[0.5, -1.0, 1.0],
		[3.0, 0.0, 1.0]
	])
	y_true = np.array([1, 0, 1])

	grad = model.calculate_gradient(y_true, X)
	m = X.shape[0] # number of data points in the batch
	y_pred = 1 / (1 + np.exp(-(X @ model.W))) # compute predicted probabilities for given X using current weights
	errors = y_pred - y_true
	expected = (1/m) * (X.T @ errors) # expected gradient calculated using the formula (1/m) * X^T @ (y_pred - y_true)

	assert grad.shape == model.W.shape, "Gradient shape does not match weight shape"
	assert np.allclose(grad, expected), "Calculated gradient does not match expected value"

def test_training():
	np.random.seed(0)
	model = LogisticRegressor(num_feats=2, learning_rate=0.1, tol=0.0, max_iter=5, batch_size=2)

	X_train = np.array([
		[0.0, 0.0],
		[1.0, 0.0],
		[0.0, 1.0],
		[1.0, 1.0]
	])
	y_train = np.array([0, 0, 0, 1])

	X_val = X_train.copy() # using the same data for validation just for testing purposes
	y_val = y_train.copy() # using the same labels for validation just for testing purposes

	initial_W = model.W.copy() # store initial weights to compare after training
	model.train_model(X_train, y_train, X_val, y_val) # train the model on the training data and validate on the validation data

	assert not np.allclose(model.W, initial_W), "Weights did not update during training"
	assert len(model.loss_hist_train) > 0, "Training loss history is empty"
	assert len(model.loss_hist_val) > 0, "Validation loss history is empty"