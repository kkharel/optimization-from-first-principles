# The Geometry of Optimization

### From a Brute-Force Search to Adam, and Why the Ranking of Optimizers Is Not a Property of the Optimizers

[![notebook](https://github.com/USERNAME/optimization-from-first-principles/actions/workflows/notebook.yml/badge.svg)](https://github.com/USERNAME/optimization-from-first-principles/actions/workflows/notebook.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Kushal Kharel**

A single self-contained notebook that builds six first-order optimizers from
nothing in NumPy, then runs a controlled experiment to test whether the story it
just told is true.

**→ [`optimization_from_first_principles.ipynb`](optimization_from_first_principles.ipynb)**

No autograd, no optimizer libraries, no framework. Every method is derived and
implemented from the definition of a derivative upward, and every numerical claim
is either computed in a cell or checkable by hand.

---

## The finding

The standard presentation of this material is a ladder: batch gradient descent,
then SGD, then mini-batch, momentum, RMSProp, and finally Adam as the
sophisticated culmination. Part III of the notebook tests that ladder and it does
not survive.

**The ranking of these six optimizers is not a property of the optimizers.** It is
a property of the curvature of the loss surface, and it reverses completely across
three regimes that differ only in how the input features were generated.

![Convergence under three curvature regimes](figures/convergence_by_regime.png)

Median over 5 seeds, learning rate tuned per method on held-out tuning seeds,
equal budget of 25 data passes. Error is the relative distance to the analytical
least-squares solution.

| Regime | κ | Best method | Its error | Batch GD | Adam |
|---|---:|---|---:|---:|---:|
| **isotropic** — independent, standardized | 1 | **Batch GD** | 1.3 × 10⁻¹⁰ | 1.3 × 10⁻¹⁰ | 7.2 × 10⁻³ |
| **anisotropic** — scales differ 30×, axis-aligned | 988 | **RMSProp** | 7.0 × 10⁻³ | 9.3 × 10⁻¹ | 7.5 × 10⁻³ |
| **correlated** — ρ = 0.95, rotated | 42 | **RMSProp** | 5.6 × 10⁻² | 4.0 × 10⁻¹ | 7.2 × 10⁻² |

Three things fall out of that table.

**On the standard test problem, the simplest method wins by seven orders of
magnitude.** At κ ≈ 1 the exact gradient points essentially at the optimum, and
batch GD is the only method here that computes the exact gradient. The
sophistication of Adam buys nothing. This matters because standardizing
independent features — unremarked boilerplate in almost every tutorial — is
exactly what drives κ to 1. The usual demonstration deletes the pathology that
motivated inventing momentum and RMSProp in the first place.

**When curvature is severe and axis-aligned, the ranking inverts.** Batch GD ends
93% away from the optimum: the stability limit set by the steep direction forces a
step size far too small for the flat one. The adaptive methods reach the target on
every seed.

**When the same ill-conditioning is rotated instead, the adaptive advantage
collapses.** RMSProp's margin over plain mini-batch SGD falls from **28× to under
2×**, on a problem whose condition number is twenty-three times *smaller*. What
changed was the orientation of the curvature, not its severity.

That last result is the point of the whole notebook, and it follows from a
one-line argument. RMSProp and Adam keep one scalar per parameter, so they are
diagonal preconditioners: they can stretch and shrink the coordinate axes, but
they cannot rotate them. A valley running diagonally through parameter space
cannot be straightened by rescaling coordinates independently.

![Three loss surfaces](figures/three_loss_surfaces.png)

Momentum has the best hit rate on the rotated surface. It averages gradient
*vectors*, so it retains the cross-coordinate structure that squaring
component-by-component throws away.

![Trajectories through a rotated valley](figures/rotated_valley_trajectories.png)

---

## What is in the notebook

70 cells, roughly 8,000 words of narrative, 12 figures, all outputs embedded.
It runs top to bottom on a fresh kernel in about 40 seconds.

| Part | Contents |
|---|---|
| **0** | Setup, and why `np.random.seed()` does not make a notebook reproducible |
| **I** | One variable: brute force → derivative → tangent line → gradient descent → the learning rate solved in closed form |
| **II** | Many parameters under a budget: SGD, mini-batch, momentum, RMSProp, Adam, each derived rather than quoted |
| **III** | The controlled experiment, and what it overturns |
| **IV** | Limitations, references, and a log of corrections |

Some things it does that a typical treatment does not:

- **Solves the learning rate exactly.** For f(x) = x², the update collapses to
  xₜ = (1−2α)ᵗ x₀, so every regime — monotone decay, one-step convergence,
  oscillation, divergence — is predicted by where a single number sits, before
  running anything.
- **Measures the finite-difference error orders** rather than asserting them, and
  shows the round-off floor where smaller h stops being better.
- **Checks the analytic gradient against a central difference** before trusting
  it, which is the standard defense against the most common silent bug in
  optimization code.
- **Unrolls the momentum recurrence** to show why exponential averaging damps
  oscillation and accelerates consistent directions with one mechanism.
- **Gets the direction of Adam's bias correction right.** The uncorrected first
  step is 3.16× too *large*, not too small — the second moment's underestimate
  dominates because it enters through a square root.
- **Catches the 1/√b noise law breaking** at large batch sizes, and fixes it with
  the finite-population correction from survey sampling.
- **Separates the two axes** that the ladder framing conflates: batch size
  controls gradient variance, the update rule controls conditioning and memory.
  They are independent choices, not rungs.

Claims that can be verified with pen and paper are marked **[verify by hand]**.

---

## Running it

```bash
git clone https://github.com/USERNAME/optimization-from-first-principles.git
cd optimization-from-first-principles

python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt

jupyter lab optimization_from_first_principles.ipynb
```

Then **Kernel → Restart & Run All**. Every figure and table in the committed
notebook regenerates identically, because every random quantity comes from a local
`np.random.default_rng(seed)` rather than the global NumPy state.

> **If GitHub will not render the notebook inline**, it is a size limit rather than
> a problem with the file. Open it on [nbviewer](https://nbviewer.org/) by pasting
> the file URL, or run it in Colab.

---

## Limitations

Stated here as well as in the notebook, because they bound what the result above
can claim.

- **The loss is quadratic and convex.** No local minima, no saddle points. The
  conditioning argument transfers to the local quadratic approximation around any
  smooth minimum, but nothing here supports claims about *finding* good minima in
  a non-convex landscape.
- **Two parameters**, chosen so the loss surface can be drawn. Adaptive methods are
  motivated partly by parameter counts in the millions, so this understates their
  value.
- **Constant learning rates, no schedule**, and only α was tuned. β₁, β₂ and batch
  size are left at defaults.
- **No second-order or non-diagonal baseline.** The natural control for the
  correlated regime is a method that *can* rotate — L-BFGS, or K-FAC-style
  approximate curvature. Its absence is the clearest gap in the design, and adding
  it would turn "diagonal methods cannot fix rotation" from an argument into a
  measurement.

---

## References

- Robbins, H. & Monro, S. (1951). A Stochastic Approximation Method. *Annals of Mathematical Statistics*.
- Polyak, B. T. (1964). Some methods of speeding up the convergence of iteration methods. — heavy-ball momentum.
- Nesterov, Y. (1983). A method for solving the convex programming problem with convergence rate O(1/k²).
- Duchi, J., Hazan, E. & Singer, Y. (2011). [Adaptive Subgradient Methods](https://jmlr.org/papers/v12/duchi11a.html). *JMLR*. — AdaGrad.
- Tieleman, T. & Hinton, G. (2012). *Lecture 6.5 — RMSProp.* Coursera.
- Kingma, D. P. & Ba, J. (2015). [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980). *ICLR*.
- Wilson, A. C. et al. (2017). [The Marginal Value of Adaptive Gradient Methods in Machine Learning](https://arxiv.org/abs/1705.08292). *NeurIPS*.
- Reddi, S., Kale, S. & Kumar, S. (2018). [On the Convergence of Adam and Beyond](https://arxiv.org/abs/1904.09237). *ICLR*.
- Bottou, L., Curtis, F. & Nocedal, J. (2018). [Optimization Methods for Large-Scale Machine Learning](https://arxiv.org/abs/1606.04838). *SIAM Review*.
- Nocedal, J. & Wright, S. (2006). *Numerical Optimization*, 2nd ed. Springer.

---

## License

MIT — see [LICENSE](LICENSE).

## Citation

```bibtex
@software{kharel_geometry_of_optimization,
  author = {Kharel, Kushal},
  title  = {The Geometry of Optimization: From a Brute-Force Search to Adam},
  year   = {2026},
  url    = {https://github.com/USERNAME/optimization-from-first-principles}
}
```
