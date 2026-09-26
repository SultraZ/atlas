# ATLAS: Adaptive Multi-Scale Framework for Structural Shape Optimization Using Extended Physics-Informed Neural Networks

> Context doc for AI-assisted development (e.g. aider). Summarizes the research proposal so the assistant understands the goal, required architecture, and — critically — what is explicitly IN and OUT of scope.

## 1. One-line summary
Build a 2D structural analysis + inverse shape design prototype in Python/PyTorch using **XPINN** (Extended Physics-Informed Neural Networks, domain decomposition) instead of a single-domain PINN, validated against FEM (FEniCS) on three benchmark problems.

## 2. Problem being solved
- FEM is accurate but expensive/slow for iterative structural design (many simulations needed).
- Vanilla single-domain PINNs struggle with complex geometry, multi-scale stress concentrations, and irregular boundaries.
- XPINNs (Hu et al., 2022) fix this by splitting the domain into subdomains, each with its own small neural network, with continuity enforced at interfaces.
- No existing work combines XPINN domain decomposition with structural **inverse shape design** — that's the novel contribution here.

## 3. What the system must do (functional requirements)
1. **Input module** — accept geometry (as SDFs or analytical boundary definitions), material properties (E, ν), loads (point/distributed/body forces), boundary conditions (Dirichlet/Neumann).
2. **XPINN solver module** — the core engine:
   - Decompose domain Ω into 2–6 non-overlapping subdomains Ωᵢ.
   - One fully-connected NN per subdomain: 4–6 hidden layers, 32–64 neurons/layer, tanh activation.
   - Three loss terms: PDE residual loss (Navier–Cauchy equilibrium), boundary condition loss, interface continuity loss (displacement + traction continuity across subdomain boundaries).
   - Optimizer: Adam, then L-BFGS fine-tuning.
   - Outputs: displacement field, strain, stress (via Hooke's law), von Mises stress.
3. **Inverse design module** — parametric shape optimization (e.g. hole radius, fillet radius, beam height) via automatic differentiation through the trained XPINN solver, minimizing compliance, without needing FEM during the optimization loop (FEM only used afterward for validation).
4. **FEM validation module** — compare against FEniCS reference solutions: L2 relative error of displacement field, L2 relative error of von Mises stress, compliance difference, max pointwise error at stress concentrations.
5. **Visualization module** — field plots (displacement, stress, von Mises), shape optimization evolution, convergence curves, ATLAS-vs-FEM side-by-side (Matplotlib/Plotly).

## 4. Governing equations (must be implemented correctly)
- Navier–Cauchy equilibrium (linear elasticity, plane stress):
  `μ∇²u + (λ+μ)∇(∇·u) + f = 0`
  where `u` = displacement vector, `μ, λ` = Lamé constants, `f` = external body force.
- Hooke's constitutive law: `σ = Dε`, where `σ` = stress, `D` = plane-stress elasticity matrix, `ε` = strain.
- Domain decomposition: `Ω = ⋃ᵢ Ωᵢ` for i = 1..N subdomains.

## 5. Benchmark problems (in order of increasing complexity)
1. **Cantilever beam** under tip load — has analytical solution; used as a sanity check for the forward solver.
2. **L-shaped bracket** with re-entrant corner — geometric complexity + sharp stress concentration; tests multi-scale domain decomposition.
3. **Plate with a circular hole** under uniaxial tension — used both for forward-accuracy testing near the hole AND for inverse shape optimization (optimize hole radius to minimize compliance).

For each benchmark: run 5 independent trials with different random initializations to check consistency.

## 6. Evaluation metrics / success criteria
- **Primary:** L² relative error of displacement field vs. FEM — target **< 5%**.
- L² relative error of von Mises stress field, especially at stress concentrations (re-entrant corner, hole boundary).
- Compliance difference between ATLAS-optimized design and FEM parametric-optimization reference — target **within 10%**.
- Convergence comparison: multi-domain XPINN vs. single-domain vanilla PINN baseline (same problems) — used to show the specific benefit of domain decomposition.

## 7. Tech stack (all open-source, ₱0 budget)
- Python
- PyTorch (network + autograd for both PDE residuals and inverse-design gradients)
- FEniCS (FEM reference/validation solver — open source)
- Matplotlib / Plotly (visualization)
- NumPy
- (Docs written in LaTeX via Overleaf — not relevant to the code itself)

## 8. Scope (what IS being built)
- 2D XPINN-based structural analysis + inverse shape design system, plane-stress linear elasticity only.
- 2–6 XPINN subdomains per benchmark, with enforced interface continuity.
- Inverse design via automatic differentiation for parametric shape variables (hole/fillet geometry).
- All forward results validated against FEM (FEniCS or similar open-source solver).
- Field visualizations (displacement, stress, von Mises stress) for every test case.

## 9. Limitations — explicitly OUT OF SCOPE (important — do not build these)
- **No 3D.** Everything is 2D plane stress only.
- **No nonlinear material behavior** (no plasticity, no hyperelasticity) and **no geometric nonlinearity** (no large deformations).
- **No dynamic/time-dependent analysis** and **no fatigue analysis** — static loading only.
- **No experimental validation.** All validation is computational (XPINN vs. FEM), never physical/lab testing.
- **No topology optimization** (density-field-based material layout) — explicitly deferred to future work. Only *parametric* shape optimization (e.g., hole radius, fillet radius) is in scope, not free-form topology.
- **Not a deployment-ready or real-time industrial tool.** This is a research prototype only.
- **No multi-material or anisotropic material support.**
- **No guarantee of global optimality** in inverse design — it's gradient-based and can land in local minima; this is expected and acceptable.

## 10. Architecture pipeline (for reference)
```
INPUT MODULE (geometry, materials, loads, BCs)
        │
        ▼
XPINN SOLVER (sub-network 1..N + interface loss)
        │
        ├──► OUTPUTS (stress, strain, displacement, compliance)
        │
        ▼
INVERSE DESIGN (shape optimization via autograd)
        │
        ▼
VALIDATION (vs FEM: L2 error, compliance)
        │
        ▼
VISUALIZATION (field plots, optimization evolution, convergence curves, ATLAS vs FEM)
```

## 11. Project context (not relevant to code, kept for reference)
This is a Grade 12 STEM (Davao City National High School) Research 1 title proposal, submitted July 2026. It is currently at the proposal/prototype stage — the described dashboards in the original document's appendices are UI/UX **concept mockups**, not built software. When implementing, prioritize getting a correct, minimal, working forward XPINN solver on the cantilever beam benchmark first (it has an analytical solution to check against) before tackling the L-bracket, the plate-with-hole, or the inverse design module.
