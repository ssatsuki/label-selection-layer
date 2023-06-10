# label-selection-layer
_Label selection layer_ is a new deep-learning-from-crowds method, which inspired by [SelectiveNet](https://arxiv.org/abs/1901.09192).

# Get started

At first, you have to install required packages using poetry. Please run the following command and install packages:

```shell
$ poetry install
```

Download some dataset to use in experiments by the following command:

```shell
$ bash download.sh
```

Preprocess some dataset by the following command:
```shell
$ poetry run python preprocess.py
```

Conduct experiments.

```shell
$ poetry run python experiments/ner_mturk_expr.py --multirun model_type=BASIC seed="range(42, 52)"
```

Set the target path created by hydra when training models and run print_results.py to print the evaluation results.

```shell
$ poetry run python print_results.py 2023-01-01/00-00-00
```
