PINN-Based Option Pricing (Black–Scholes PDE)
🧠 Overview

This project implements a Physics-Informed Neural Network (PINN) to solve the Black–Scholes Partial Differential Equation (PDE) for European option pricing.
Instead of relying on traditional numerical methods, the model learns the solution by enforcing the governing PDE and financial constraints directly within the loss function.

🚀 Key Features
📊 Solves Black–Scholes PDE using deep learning
🧮 No grid-based discretization required
⚙️ Physics-informed loss (PDE + boundary + terminal conditions)
📈 Comparison with analytical Black–Scholes solution
🔁 Hybrid optimization (Adam + L-BFGS)
🌐 Generalization across different volatility regimes
⚙️ Problem Formulation

The model approximates the option pricing function:

Input: Asset Price (S), Time (t), Volatility (σ)
Output: Option Price 
V(S,t)

The Black–Scholes equation is enforced through the loss function along with:

Terminal payoff condition
Boundary conditions
🏗️ Project Workflow
1. Financial Setup
Define parameters: Strike price (K), risk-free rate (r), maturity (T), volatility (σ)
2. Data Generation (Physics-Based)
Interior Points: Enforce PDE
Terminal Points: Enforce payoff 
V(S,T)=max(S−K,0)
Boundary Points: Ensure financial constraints
3. Model Architecture
Fully connected neural network
Input: (S, t, σ)
Output: Option price
4. Training Process
Forward pass to compute predictions
Automatic differentiation for PDE derivatives
Loss computation:
PDE residual loss
Terminal loss
Boundary loss
5. Optimization
Adam optimizer for initial training
L-BFGS optimizer for fine-tuning
6. Evaluation
Compare predictions with analytical Black–Scholes solution
Analyze errors across different scenarios
7. Visualization
Option price vs asset price
Time evolution
Volatility sensitivity
3D price surfaces
📊 Results
Accurate approximation of Black–Scholes solution
Stable performance across multiple volatility regimes
Smooth continuous pricing function learned by the network
🧩 Tech Stack
Python
PyTorch
NumPy
Matplotlib
🔮 Future Improvements
Extend to American options
Incorporate stochastic volatility models (Heston)
Multi-asset option pricing
Calibration using real market data
💡 Key Insight

Unlike traditional supervised learning, this model does not require labeled data.
It learns by minimizing the violation of the governing PDE and financial constraints.
