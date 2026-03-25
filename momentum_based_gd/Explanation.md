## Gradient Descent with Momentum

This notebook demonstrates how **Momentum-based Gradient Descent** helps optimization move faster and smoother compared to **Batch Gradient Descent (BGD)**.

---

### 1) Loss Function

We use a simple quadratic loss function:

\[
J(x, y) = x^2 + 10y^2
\]
---

### 2) Gradient of the Loss

To minimize the loss, we calculate its gradient:

\[
\frac{\partial J}{\partial x} = 2x
\]

\[
\frac{\partial J}{\partial y} = 20y
\]

So the gradient vector is:

\[
\nabla J(x, y) = (2x,\; 20y)
\]

The gradient shows the direction of steepest increase, so we move in the opposite direction to reduce the loss.

---

### 3) Batch Gradient Descent

In Batch Gradient Descent, the parameters are updated using only the current gradient:

\[
x_{t+1} = x_t - \eta \cdot \frac{\partial J}{\partial x}
\]

\[
y_{t+1} = y_t - \eta \cdot \frac{\partial J}{\partial y}
\]

Here, \(\eta\) is the learning rate.

BGD updates the parameters directly based on the gradient, but it can sometimes oscillate, especially when the loss surface is steep in one direction and flat in another.

---

### 4) Gradient Descent with Momentum

Momentum improves gradient descent by adding a moving average of past gradients.

First, we update the velocity:

\[
v_t = \beta v_{t-1} + (1 - \beta)\nabla J(x_t, y_t)
\]

Then we update the parameters:

\[
x_{t+1} = x_t - \eta \cdot v_t^{(x)}
\]

\[
y_{t+1} = y_t - \eta \cdot v_t^{(y)}
\]

Where:

- \(\beta\) is the momentum coefficient (usually close to 1)
- \(v_t\) is the velocity term
- \(\eta\) is the learning rate

Momentum helps the optimizer:

- Move faster in the right direction  
- Reduce zig-zag oscillations  
- Smooth the path toward the minimum  

---

### 5) Why Momentum Works Better

In a curved loss surface, plain gradient descent may keep bouncing from side to side.  
Momentum remembers previous updates, so it builds speed in consistent directions and reduces unnecessary oscillations.

That is why momentum usually reaches the minimum more smoothly than standard BGD.

---

### 6) What This Notebook Shows

This notebook compares:

- Batch Gradient Descent  
- Gradient Descent with Momentum  

It plots:

- The optimization path on the loss surface  
- The loss value over epochs  

From the plots, you can observe that momentum generally gives a smoother and faster path toward the minimum.

---

### 7) Key Takeaway

Momentum-based gradient descent is an improved version of standard gradient descent.  
It uses past updates to accelerate learning and reduce oscillations, making optimization more stable and efficient.
