START
  ↓
[1] Define Financial Setup
  ├── Strike Price (K)
  ├── Risk-Free Rate (r)
  ├── Time to Maturity (T)
  └── Volatility Range (σ)
  ↓
  (These define the Black–Scholes environment)

  ↓
[2] Generate Training Data (Crucial Step)
  ├── Interior Points (Collocation Points)
  │     • S_interior → Asset prices sampled in domain
  │     • t_interior → Time sampled in (0, T)
  │     • σ_interior → Volatility samples
  │
  ├── Terminal Points (Maturity Condition)
  │     • S_terminal
  │     • t_terminal = T
  │     • Enforces payoff:
  │         max(S - K, 0)
  │
  ├── Boundary Points
  │     • S → 0  (Option value → 0)
  │     • S → large (Option behaves ~ S - K)
  │
  ↓
  (Dataset is physics-driven, NOT labeled data)

  ↓
[3] Define Neural Network (PINN)
  ├── Input Layer:
  │     (S, t, σ)
  │
  ├── Hidden Layers:
  │     Fully connected layers with activation (tanh/relu)
  │
  ├── Output Layer:
  │     V(S, t) → Option price
  │
  ↓
  (Model approximates continuous pricing function)

  ↓
[4] Forward Pass
  ├── Pass all inputs through network
  └── Get predicted prices V̂(S, t)

  ↓
[5] Compute Derivatives using Autograd
  ├── ∂V/∂t
  ├── ∂V/∂S
  └── ∂²V/∂S²
  ↓
  (Required to enforce PDE)

  ↓
[6] Construct Loss Function (Core of PINN)
  ├── [A] PDE Residual Loss
  │     • Enforces:
  │       ∂V/∂t + (1/2)σ²S² ∂²V/∂S² + rS ∂V/∂S - rV = 0
  │     • Computed at interior points
  │
  ├── [B] Terminal Loss
  │     • Ensures:
  │       V(S, T) = max(S - K, 0)
  │
  ├── [C] Boundary Loss
  │     • S → 0 ⇒ V ≈ 0
  │     • S → ∞ ⇒ V ≈ S - K
  │
  ├── Total Loss:
  │     Loss = PDE + Terminal + Boundary
  │
  ↓
  (This replaces traditional supervised learning)

  ↓
[7] Backpropagation
  ├── Compute gradients of loss
  └── Update network weights

  ↓
[8] Optimization Strategy
  ├── Phase 1: Adam Optimizer
  │     • Fast initial convergence
  │
  ├── Phase 2: L-BFGS Optimizer
  │     • Fine-tuning for precision
  │
  ↓
  (Hybrid optimization improves stability)

  ↓
[9] Model Convergence Check
  ├── Loss stabilization
  ├── PDE residual minimization
  └── Visual sanity checks

  ↓
[10] Analytical Benchmarking
  ├── Compute exact Black–Scholes solution
  ├── Compare:
  │     • Predicted vs Exact
  │     • Error metrics
  ↓

[11] Visualization & Analysis
  ├── Option price vs Asset Price (S)
  ├── Time evolution plots
  ├── Volatility sensitivity (σ)
  ├── Surface plots (S, t → V)
  ↓

[12] Insights & Validation
  ├── Accuracy across regimes
  ├── Stability of training
  └── Generalization capability

  ↓
END
