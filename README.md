# PoseGen: Self-Supervised Pose-Conditioned Image Generation

## Getting Started

* Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
* Create and activate conda environment.

```shell
conda env create -f conda_env.yml
conda activate posegen
```

> NOTE: PyTorch dependency specified in `environment/conda_env.yml` uses CUDA 11.1. If CUDA 11.1 is unsupported on your environment, please install PyTorch separately by following the [official instructions](https://pytorch.org).

### Datasets
* Download Stanford Cars dataset.
```shell
cd data_dir
sh {PoseGen DIR}/src/misc/stanford_cars_dataset_downloader.sh
```

* Download Tesla
```shell
TODO
```

### Experiments
```shell
python experiment.py
```
