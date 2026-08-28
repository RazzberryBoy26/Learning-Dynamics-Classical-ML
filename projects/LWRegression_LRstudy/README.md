# Mathematical Analysis of Gradient Descent Dynamics & Learning Rate Optimization

This document provides a rigorous mathematical breakdown of gradient descent convergence, optimal learning rates, and computational complexity for **Standard Linear Regression** and **Locally-Weighted Regression (LWR)**.

---

## 📌 Overview of the Work

The accompanying document (`.tex` / `.pdf`) presents an analytical, optimization-theoretic framework addressing key questions in first-order optimization for regression models:

1. **Exact Convergence Bounds**: Why and when does batch gradient descent converge to the analytical Ordinary Least Squares (OLS) solution?
2. **Optimal Learning Rate ($\eta_{\text{opt}}$)**: Deriving the step size that minimizes the iteration count $N(\eta)$ to reach an $\varepsilon$-neighborhood of the global minimum via minimax spectral analysis.
3. **Locally-Weighted Regression (LWR) Extension**: Formulating the query-weighted loss function, gradient, and Hessian $H(\theta) = \frac{1}{m}X^T W X$.
4. **Hessian-Free Convergence Guarantee ($\eta_{\text{guarantee}}$)**: Deriving an analytical lower bound for the maximum allowable learning rate using matrix outer-product spectral properties, bypassing expensive Hessian and eigenvalue computations.
5. **Computational Complexity Trade-off**: Asymptotic runtime analysis comparing $\eta_{\text{guarantee}}$ vs. $\eta_{\text{opt}}$ across sample-heavy ($m \gg n$) and feature-heavy ($n \gg m$) regimes.

---

## 🔬 Key Mathematical Derivations

### 1. Linear Regression Convergence Dynamics
* **Loss Function**: $J(\Theta) = \frac{1}{2m} (X\Theta - Y)^T (X\Theta - Y)$
* **Error Recurrence**: $e^{(t+1)} = \left(I - \frac{\eta}{m} X^T X\right) e^{(t)} = (I - \eta H)e^{(t)}$
* **Spectral Convergence Condition**:
  $$\rho(I - \eta H) < 1 \implies 0 < \eta < \frac{2}{\lambda_{\max}(H)}$$
* **Optimal Step Size Derivation**:
  By minimizing the worst-case spectral radius via minimax optimization of the iteration bound:
  $$\eta_{\text{opt}} = \frac{2}{\lambda_{\max}(H) + \lambda_{\min}(H)}$$

---

### 2. Locally-Weighted Regression (LWR) & $\eta_{\text{guarantee}}$
* **Objective Function**: $J(\theta) = \frac{1}{2m}(X\theta - Y)^T W(X\theta - Y)$ where $W = \operatorname{diag}(\omega^{(1)}, \dots, \omega^{(m)})$ and $\omega^{(i)} = \exp\left(-\frac{\|x^{(i)} - x_q\|_2^2}{2\tau^2}\right)$.
* **Hessian Outer-Product Decomposition**:
  $$H = \frac{1}{m} \sum_{p=1}^m \omega^{(p)} x^{(p)} (x^{(p)})^T$$
* **Spectral Bound using Subadditivity & Rank-1 Property** ($\lambda_{\max}(u u^T) = \|u\|_2^2$):
  $$\lambda_{\max}(H) \le \frac{1}{m} \sum_{p=1}^m \omega^{(p)} \|x^{(p)}\|_2^2$$
* **Guaranteed Safe Step Size**:
  $$\eta_{\text{guarantee}} = \frac{m}{\sum_{p=1}^m \omega^{(p)} \|x^{(p)}\|_2^2} \le \frac{2}{\lambda_{\max}(H)}$$
  *(Guarantees convergence without ever computing matrix products or eigenvalues)*.

---

## ⚡ Computational Complexity Comparison

| Parameter / Metric | Asymptotic Time Complexity | Operations Involved | Best Suited For |
| :--- | :--- | :--- | :--- |
| **$\eta_{\text{guarantee}}$** | $\Theta(mn)$ | Vector norms + scalar weighting | High-dimensional regimes ($n \gg m$) |
| **$\eta_{\text{opt}}$** | $\Theta(mn^2 + n^3)$ | Hessian assembly + Eigendecomposition | Overdetermined regimes ($m \gg n$) |

---

## 📊 Empirical Loss Curve Validation Summary

The theoretical derivations yield four distinct empirical regimes when running gradient descent:
* $\eta = \eta_{\text{opt}}$: **Fastest convergence rate** (minimal iterations to reach $\varepsilon$-tolerance).
* $\eta = \eta_{\text{guarantee}}$: **Guaranteed convergence** with zero Hessian matrix computation overhead.
* $\eta \sim U\left(0, \frac{2}{\lambda_{\max}}\right)$: **Stable convergence** (sub-optimal speed).
* $\eta > \frac{2}{\lambda_{\max}}$: **Divergence** (spectral radius strictly exceeds unity).