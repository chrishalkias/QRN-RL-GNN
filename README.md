# Quantum Repeater Network with RL and GNNs


>[!WARNING]
>This project is still under construction and not operational. The code only offers limited usability far below the scope of the full project. When all is done, after the shampagne :beers:, remove this banner and declare victory.

## About the project
This is the repository for my MSc thesis on studying entanglement distribution in ***quantum repeater networks*** using Reinforcement Learning. The goal of this project is to examine how Reinforcement Learning agents can learn efficient policies for enanglement distribution in 1D repeater chains. The aim is that given a set of physical parameters $C_n$ an agent can establish a policy $\pi_n$ through the mapping:
$$\pi(C_n; NN) = \pi,  C_N = (n, p_e, p_s, \tau)$$
where, $NN$ encodes the neural network parameters. Afterwards, this policy is to be evaluated on a different set of physical parameters $C_{n'}$ and hopefuly we can find an agent that can learn optimal policies that can transfer to systems of different sizes.

Additionally, there is a simple tabular puzzle game that translates the system into a more interpretable version that can also be played by humans!

>[!note]
>The game has some yet to be fixed bugs, described in the initial screen


## About the code
The repository's main files are located into the `src` folder. The `game` folder contains the files of the quantum game and is independent of the rest of the code structure. The hope is that eventually it will grow up to have a repository of its own. The `logs` folder consists of all of the programs output including plots and text files.

the full code structure can be found below:

```
.
├── game
│   ├── __init__.py
│   ├── game.py
│   ├── play.py
│   └── README.md
├── src
│   ├── __init__.py
│   ├── cnn_environment.py
│   ├── cnn_main.py
│   ├── gnn_environment.py
│   ├── gnn_main.py
│   ├── logs
│   ├── mlp_gym_env.py
│   ├── models.py
│   ├── plot_config.py
│   ├── repeaters.py
│   └── sandbox.ipynb
├── logs
│   ├── models
│   │   ├── model_summary.txt
│   │   └── model.pth
│   ├── plots
│   │   ├── test_alternating.png
│   │   ├── test_random.png
│   │   ├── test_trained.png
│   │   └── train_plots.png
│   └── textfiles
│       ├── alternating_test_output.txt
│       ├── random_test_output.txt
│       └── trained_test_output.txt
├── example.ipynb
├── LICENSE
├── pyproject.toml
├── README.md
├── requirements.txt
└── test.py
```

## Instalation
  You can install the code by:
```

git clone https://github.com/chrishalkias/QRN-RL-GNN

```

## Additional Information
The physical system used is Quantum repeaters. Big picture and outlook YouTube video from [QuTech](https://www.youtube.com/watch?v=9iCFH9Fk184) and [Qunnect (animation)](https://www.youtube.com/watch?v=3_oqkFO4f-A). The project has a similar scope to [Haldar et al.](https://arxiv.org/abs/2303.00777) but the idea is to either use more complex geometries (repeater networks insdead of repeater chains) and more state of the art architectures for the RL agent ([Graph convolutional neural networks](https://arxiv.org/pdf/1609.02907), [Transformers](https://arxiv.org/abs/1706.03762), [Graph Attention networks](https://arxiv.org/abs/1710.10903)) or extend to scale (n) invariant models.



  

