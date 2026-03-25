## Adam Optimizer

Adam (Adaptive Moment Estimation) is an optimization algorithm that combines the ideas of **Momentum** and **Adaptive Learning Rates** to make training faster and more stable.

---

### 1) Goal

We want to minimize a loss function:

\[
J(\theta)
\]

At each step, we compute the gradient:

\[
g_t = \nabla J(\theta_t)
\]

This gradient tells us the direction in which the loss increases, so we move in the opposite direction.

---

### 2) First Moment (Momentum Idea)

Adam keeps a running average of past gradients:

\[
m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t
\]

- This is similar to **momentum**
- Instead of reacting only to the current gradient, it remembers past gradients
- This helps reduce noisy updates and smooths the direction of movement

---

### 3) Second Moment (Adaptive Learning)

Adam also keeps track of squared gradients:

\[
v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2
\]

- This measures how large the gradients are
- If gradients are large → we take smaller steps  
- If gradients are small → we take larger steps  

So the learning rate becomes **adaptive for each parameter**

---

### 4) Bias Correction

At the beginning, both \(m_t\) and \(v_t\) start from zero, so they are biased.

To fix this:

\[
\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad
\hat{v}_t = \frac{v_t}{1 - \beta_2^t}
\]

This makes the estimates more accurate, especially in early iterations.

---

### 5) Final Update Rule

\[
\theta_{t+1} = \theta_t - \eta \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
\]

Where:

- \(\eta\) = learning rate  
- \(\beta_1\) controls momentum (usually 0.9)  
- \(\beta_2\) controls scaling (usually 0.999)  
- \(\epsilon\) prevents division by zero  

---

### 6) Intuition (Very Important)

Adam does two things at once:

- Uses **momentum** → moves smoothly in the right direction  
- Uses **adaptive scaling** → adjusts step size automatically  

This makes optimization:
- faster  
- more stable  
- less sensitive to hyperparameters  

---

### 7) Comparison with Other Optimizers

**Gradient Descent**
- Same step size everywhere  
- Slow and oscillates  

**Momentum**
- Reduces oscillations  
- Moves faster in consistent directions  
- Still uses fixed learning rate  

**Adam**
- Adjusts learning rate for each parameter  
- Converges faster  
- Produces smoother optimization path  

---

### 8) Observations from Your Plots

- Loss decreases much faster with Adam  
- The path to the minimum is smoother  
- Fewer iterations are needed compared to GD and Momentum  

---

### 9) Key Takeaway

Adam combines the strengths of Momentum and adaptive methods:

\[
\text{Adam} = \text{Momentum} + \text{Adaptive Learning Rate}
\]

This is why it is one of the most widely used optimizers in deep learning.
