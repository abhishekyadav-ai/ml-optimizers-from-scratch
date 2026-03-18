# Stochastic Gradient Descent (SGD) From Scratch

## Objective

The goal of this implementation is to understand how Stochastic Gradient Descent (SGD) differs 
from Batch Gradient Descent and how it improves efficiency for large datasets.

SGD is widely used in machine learning and deep learning to optimize model parameters, 
especially when working with large-scale data.


In Batch Gradient Descent, parameter updates are computed using the entire dataset.

This becomes computationally expensive when:

- the dataset is very large  
- the number of parameters increases  

For example:

If a model has many parameters and the dataset contains millions of data points, 
each update step requires processing the entire dataset, making training slow.

## Key Idea

Stochastic Gradient Descent addresses this problem by updating parameters 
using **one data point at a time** instead of the full dataset.

Instead of computing:

gradient over all samples

SGD computes:

gradient using a single randomly selected sample

## Model

We use linear regression as the model:

y = mx + c

Where:

- m is the slope  
- c is the intercept

## Loss Function

The loss function used is Mean Squared Error (MSE):

L = (1/n) Σ (y - ŷ)²

The objective is to minimize this loss.

## How SGD Works

The training process follows these steps:

1. Shuffle the dataset at the beginning of each epoch  
2. Iterate through each data point  
3. For each sample:
   - compute prediction  
   - calculate error  
   - compute gradient  
   - update parameters immediately  

This results in many small updates instead of one large update.

## Parameter Update Rule

For each data point (xᵢ, yᵢ):

∂L/∂m = -2 xᵢ (yᵢ - ŷᵢ)  
∂L/∂c = -2 (yᵢ - ŷᵢ)  

Update:

m = m - α * ∂L/∂m  
c = c - α * ∂L/∂c  

Where α is the learning rate.


## Implementation Insight

One important part of the implementation is:

X.T.dot(errors)

This operation represents how each feature contributes to the overall error signal.

It aggregates the influence of input features on the loss, which is then used to update model parameters.


## Conclusion

Stochastic Gradient Descent improves efficiency by performing frequent updates using small portions of data.

Although it introduces noise in the optimization process, it significantly 
speeds up learning and scales well for large datasets.

Understanding SGD helps build intuition for more advanced optimizers such as Momentum, RMSProp, and Adam.
