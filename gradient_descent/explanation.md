# Gradient Descent Implementation From Scratch

## Objective

The goal of this implementation is to understand how machine learning models learn parameters by minimizing a loss function using gradient descent.

Instead of using libraries such as Scikit-learn or PyTorch, the algorithm is implemented manually using Python and NumPy to better understand the underlying optimization process.

---

## Dataset

A simple synthetic dataset is generated using the relationship:

y = 2x + 1 + noise

The dataset is created using:

- `np.linspace()` to generate evenly spaced values for x
- `np.random.normal()` to add random noise

This produces data points that roughly follow a straight line but include some variation.

The objective of the model is to learn the best fitting line for these data points.

---

## Model

The model used is **linear regression**.

The relationship between input and output is defined as:

y = mx + c

Where:

- **m** is the slope of the line
- **c** is the intercept

The parameters start with initial guesses:

m = 0

c = 0


The goal of gradient descent is to iteratively update these parameters so the line best fits the data.

---

## Loss Function

To measure how well the model fits the data, the **Mean Squared Error (MSE)** is used.

MSE calculates the average squared difference between actual values and predicted values.

L = (1/n) Σ (y - ŷ)²

Where:

- y = actual value
- ŷ = predicted value
- n = number of data points

The objective of training is to **minimize this loss**.

---

## Gradient Descent Idea

Gradient descent updates model parameters in the direction that reduces the loss.

The gradients of the loss function with respect to the parameters are:

∂L/∂m = (-2/n) Σ X (y - ŷ)

∂L/∂c = (-2/n) Σ (y - ŷ)

These gradients indicate how much the parameters should change to reduce the error.

---

## Parameter Update Rule

Parameters are updated using the following rule:

m = m - α * ∂L/∂m  
c = c - α * ∂L/∂c  

Where:

- **α (alpha)** is the learning rate.

The learning rate determines how large each update step is during optimization.

---

## Training Process

The algorithm performs the following steps repeatedly:

1. Compute predictions using the current parameters.
2. Calculate gradients of the loss function.
3. Update the parameters using gradient descent.
4. Calculate the loss.
5. Store the loss value for visualization.

This process is repeated for multiple iterations.

Over time:

- the loss decreases
- the fitted line moves closer to the data points.

---

## Visualizations

Two visualizations are produced in this implementation:

**1. Dataset and Fitted Line**

This plot shows the original data points and the regression line learned by gradient descent.

**2. Loss vs Iterations**

This plot shows how the loss decreases as the algorithm updates the parameters during training.

---

## Conclusion

This implementation demonstrates how gradient descent optimizes model parameters by iteratively reducing the loss function.

Understanding this process helps build intuition about how machine learning models learn before using higher-level machine learning libraries.
