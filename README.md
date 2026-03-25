# ML Optimizers From Scratch

This repository contains from-scratch implementations of core optimization algorithms used in machine learning and deep learning.

The focus is not just implementation, but understanding **how optimization behaves** — through mathematical formulation and visualization of convergence.

---

## Implemented Optimizers

- Gradient Descent (Batch GD)
- Stochastic Gradient Descent (SGD)
- Momentum-based Gradient Descent
- Adam Optimizer

---

## What This Project Shows

- How optimization reduces loss over iterations  
- Effect of learning rate on convergence  
- Oscillation vs smooth convergence (GD vs Momentum)  
- Faster convergence using adaptive methods (Adam)  

---

## Tech Stack

- Python  
- NumPy  
- Matplotlib  

---

## Project Structure

- `gradient_descent/` → Batch Gradient Descent  
- `stochastic_gradient_descent/` → SGD vs GD comparison  
- `momentum_based_gd/` → Momentum vs GD  
- `adam/` → Adam optimizer  

---

## Why This Project

Libraries like PyTorch and TensorFlow abstract away optimization.

This project rebuilds these algorithms from scratch to understand:
- how gradients drive learning  
- how different optimizers affect convergence  
- why advanced optimizers outperform batch gradient descent  

---
