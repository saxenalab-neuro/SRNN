# SRNN — Switching Recurrent Neural Networks

[![Paper](https://img.shields.io/badge/paper-NeurIPS%202024-blue)](https://openreview.net/pdf?id=zb8jLAh2VN)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Official implementation of **Switching Recurrent Neural Networks (SRNNs)** — a model class that combines the expressivity of RNNs with the interpretable regime structure of switching state-space models, enabling inference of discrete dynamical regimes underlying neural activity.

> **Inference of Neural Dynamics Using Switching Recurrent Neural Networks**
> Yongxu Zhang, Shreya Saxena.
> *Advances in Neural Information Processing Systems (NeurIPS)*, 2024.
> [[paper]](https://openreview.net/pdf?id=zb8jLAh2VN)

![Switching Recurrent Neural Networks](data/srnn.png)

## What's in this repo

| Path | What it contains |
|------|------------------|
| `SRNN/` | Core SRNN model implementation (`model_srnn.py`), inference network, EM training loop, and utilities. |
| `ssm/` | Vendored state-space-model helpers used by the demos. |
| `0_demo_lorenz*.ipynb` | Walk-through notebooks fitting an SRNN to a Lorenz attractor under different initializations. |
| `1_demo_lorenz.py` | Same Lorenz demo as a runnable script. |
| `1_demo_analysis.ipynb` | Post-fit analysis of inferred switching regimes. |
| `environment_srnn.yml` | Conda environment specification. |

## Installation

```bash
git clone https://github.com/saxenalab-neuro/SRNN.git
cd SRNN
conda env create --file environment_srnn.yml
conda activate SwitchingRNN
```

If the environment build fails on your platform, `pip install` any reported missing packages — they are all common scientific-Python dependencies.

## Quick start

Run the Lorenz-attractor demo end-to-end:

```bash
python 1_demo_lorenz.py
```

Or work through the demo interactively:

```bash
jupyter notebook 0_demo_lorenz.ipynb
```

For analysis of a fitted model (inferred regimes, switching dynamics, reconstructions), open `1_demo_analysis.ipynb`.

## Citation

If you use SRNN in your research, please cite:

```bibtex
@inproceedings{zhang2024srnn,
  title     = {Inference of Neural Dynamics Using Switching Recurrent Neural Networks},
  author    = {Zhang, Yongxu and Saxena, Shreya},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2024},
  url       = {https://openreview.net/pdf?id=zb8jLAh2VN}
}
```

A `CITATION.cff` is included so you can also click *"Cite this repository"* on GitHub.

## License

[MIT](LICENSE) © Saxena Lab, Yale University.

## Acknowledgements

Developed in the [Saxena Lab for Neural Control](https://www.saxenalab.org) at Yale University. See the [lab GitHub organization](https://github.com/saxenalab-neuro) for related tools (mRNNTorch, RNNToolkit, muSim).
