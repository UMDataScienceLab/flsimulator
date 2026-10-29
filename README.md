# FLsimulator

This repository intends to use the simplest code to simulate federated learning fast.


## Getting Started

- Change the params in main.py
- Type ./run.sh in shell.
- Results are generated in logs/ folder.


## Files

The implementation includes (multi-processing) fedavg, fedensemble. They are implemented in server.py and client.py

## Contributing

Contributions are welcome! If you're interested in improving the multi-processing, or adding new algorithms, feel free to add a new pull request.

## Citation

This simulator implements Fed-ensemble ([paper](https://doi.org/10.1109/TASE.2023.3269639), IEEE Transactions on Automation Science and Engineering, 2024). If you use it, please cite:

```bibtex
@article{shi2024fedensemble,
  title={Fed-ensemble: Ensemble Models in Federated Learning for Improved Generalization and Uncertainty Quantification},
  author={Shi, Naichen and Lai, Fan and Al Kontar, Raed and Chowdhury, Mosharaf},
  journal={IEEE Transactions on Automation Science and Engineering},
  volume={21},
  number={3},
  pages={2792--2803},
  year={2024},
  doi={10.1109/TASE.2023.3269639}
}
```
