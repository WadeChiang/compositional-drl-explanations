# compositional-drl-explanations

This repository contains the official implementation of the paper:  
**Compositional Concept-Based Neuron-Level Explanations for Deep Reinforcement Learning**  
Accepted at PAKDD 2026 (DSFA Special Session)
## Running Experiments

```bash
cd exp/
python analyze_blackjack.py
python analyze_lunar.py
```

visualization check and perturbations can be found in `exp/display.ipynb`, folders used in paper are:

```
save/Blackjack2-DQN64    # Blackjack experiment results
save/LunarLander-DQN64   # LunarLander experiment results
save/LL_hyper/*          # LunarLander hyperparameter tuning results
```
