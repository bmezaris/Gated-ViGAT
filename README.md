# Gated-ViGAT: Efficient bottom-up event recognition and explanation using a new frame selection policy and gating mechanism

This repository hosts the code and data for our paper: N. Gkalelis, D. Daskalakis, V. Mezaris, "Gated-ViGAT: Efficient bottom-up event recognition and explanation using a new frame selection policy and gating mechanism", IEEE International Symposium on Multimedia (ISM)

## Gated-ViGAT scripts, and traning and evaluation procedures

### Code requirements

* numpy
* scikit-learn
* PyTorch

### Training

Ιn order to train Gated-ViGAT on any video-dataset, the corresponding ViGAT model should be present. The models for ActivityNet and miniKinetics are available inside weights folder. To train a new gate , run 
```
python train_gate.py weights/<vigat model>.pt --dataset_root <dataset dir> --dataset [<actnet|minikinetics>]
```

The training parameters can be modified by specifying the appropriate command line arguments. For more information, run ```python train_gate.py --help```.

### Evaluation

To evaluate a gate, run
```
python evaluation_gate.py weights/<vigat model>.pt weights/<model name>.pt --dataset_root <dataset dir> --dataset [<actnet|minikinetics>]
```
Again, the evaluation parameters can be modified by specifying the appropriate command line arguments. For more information, run ```python evaluation_gate.py --help```.

## Acknowledgements

This work was supported by the EU Horizon 2020 programme under grant agreement 101021866 (CRiTERIA);

## References

[1] B. G. Fabian Caba Heilbron, Victor Escorcia and J. C. Niebles. ActivityNet: A large-scale video benchmark for human activity understanding. In Proc. IEEE CVPR, 2015, pp. 961–970.
