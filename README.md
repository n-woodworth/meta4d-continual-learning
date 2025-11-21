Meta4D: Homeostatic Topology Regulation for Continual Learning

A self-evolving Transformer architecture that mitigates catastrophic forgetting via Lyapunov-guided dynamic sparsity.


Abstract

Standard neural networks suffer from Catastrophic Forgetting: learning a new task ($T_B$) destroys the weights optimized for a previous task ($T_A$). Meta4D solves this not by constraining weights (EWC), but by evolving the Topology.

By treating the network as a dynamical system regulated by a Lyapunov energy function, Meta4D achieves Positive Backward Transfer (+12% BWT) on the Permuted MNIST benchmark. It physically grows new "lobes" for new tasks while chemically latching old memories, mimicking biological neurogenesis and synaptic consolidation.

## Key Results (Empirical Proof)

| Metric                      | Baseline (Static) | Meta4D | Delta.                 |
|:----------------------------|:--------|:-----------|:-----------------------      |
| **Final Loss (Task 1)**     | 0.0987            | 0.0092 |-90%(Order of Magnitude)|
| **Backward Transfer (BWT)** | +0.0258           | +0.1236| +379%                  |
| **Active Params (Biomass)** | 100%              | ~35%   | 3x Efficiency          |

Conclusion: The architecture does not just "remember" Task A; it uses the structural complexity grown during Task B to better solve Task A (Synergistic Learning).

The Physics of Meta4D

1. Thermodynamics (Lyapunov Control)

The system minimizes a composite energy function $V_t$:


$$V_t = \mathcal{L}_{task} + \lambda \cdot (\text{Biomass}_t - C_{target})^2$$

High Energy (Stress): Triggers Neurogenesis (SPAWN). The network grows new connections to dissipate error.

Low Energy (Stability): Triggers Apoptosis (PRUNE). The network aggressively culls weak connections to minimize metabolic cost.

2. Hebbian Structural Learning

Instead of random pruning, Meta4D uses Gradient-Guided Spawning.

It calculates the "Virtual Gradient" ($\frac{\partial L}{\partial w}$) for dormant blocks.

It wakes up only the blocks that maximally oppose the error vector.

Result: "Neurons that need to fire together, wire together."

3. The "Sleep" Cycle (Rolling Defragmentation)

To maintain 100% uptime, the system performs Rolling Consolidation. It locks 1% of the network blocks per step to perform aggressive optimization/pruning, rotating through the entire brain every epoch. This mimics the function of sleep without the downtime.

Quick Start

Prerequisites

Python 3.8+

PyTorch 2.0+ (CUDA recommended for Block Sparsity speedups)

Installation

git clone [https://github.com/n-woodworth/meta4d-continual-learning.git](https://github.com/n-woodworth/meta4d-continual-learning.git)
cd meta4d-continual-learning
pip install -r requirements.txt


Reproduction (The Proof)

To replicate the Permuted MNIST benchmark and generate the BWT table:

python comparative_meta4d_metrics.py


To run the Transformer on sequential text (TinyShakespeare):

python transformer_continual_learning.py


Citation

If you use this architecture in your research, please cite:

@article{woodworth2025meta4d,
  title={Meta4D: Homeostatic Regulation of Transformer Topology Mitigates Catastrophic Forgetting},
  author={Woodworth, Nicholas A.},
  year={2025},
  publisher={GitHub},
  journal={GitHub Repository},
  howpublished={\url{[https://github.com/n-woodworth/meta4d-continual-learning](https://github.com/n-woodworth/meta4d-continual-learning)}}
}


License

This project is licensed under the MIT License - see the LICENSE
