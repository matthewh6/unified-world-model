# Unified World Models: Coupling Video and Action Diffusion for Pretraining on Large Robotic Datasets

####  [[Website]](https://weirdlabuw.github.io/uwm/) [[Paper]](https://arxiv.org/abs/2504.02792) [[Talk]](https://www.youtube.com/watch?v=WwPRxBbZ4kw)

[Chuning Zhu<sup>1</sup>](https://homes.cs.washington.edu/~zchuning/), [Raymond Yu<sup>1</sup>](https://raymondyu5.github.io/), [Siyuan Feng<sup>2</sup>](https://www.cs.cmu.edu/~sfeng/), [Benjamin Burchfiel<sup>2</sup>](https://scholar.google.com/citations?user=eGoTK1YAAAAJ&hl=en), [Paarth Shah<sup>2</sup>](https://www.paarthshah.me/about), [Abhishek Gupta<sup>1</sup>](https://homes.cs.washington.edu/~abhgupta/)<br/>

<sup>1</sup>University of Washington <sup>2</sup>Toyota Research Institute

This is a PyTorch implementation of Unified World Model (UWM). UWM combines action diffusion and video diffusion to enable scalable pretraining on large, heterogeneous robotics datasets.


## Code structure
- `configs` contains the config files for pretraining and finetuning experiments.
- `datasets` contains the dataset classes for DROID, Robomimic, and LIBERO. We standardize the datasets to use compressed [Zarr](https://zarr.readthedocs.io/en/stable/) buffers. 
- `environments` contains wrappers to interface with Robomimic and LIBERO environments.
- `experiments` contains training and evaluation scripts.
- `models` contains model definitions for UWM and baselines.
- `scripts` contrains bash scripts for DROID experiments.


## Setup
Install requirements using 
```
pip install -r requirements
``` 

Add current directory to PYTHONPATH
```
export PYTHONPATH=.
```

## Robomimic Single-Task Experiments

To run a Robomimic single-task experiment, install the [robomimic](https://github.com/ARISE-Initiative/robomimic) dataset. Then, update the hdf5 and buffer paths in the config file (e.g. `robomimic_cap_ph.yaml`) and run
```
python experiments/uwm/train_robomimic.py --config-name train_uwm_robomimic.yaml dataset=robomimic_can_ph exp_id=singletask
```
Note that this will create a Zarr compressed buffer at the `buffer_path` specified in the config file.

## LIBERO Pretraining / Finetuning Experiments
The LIBERO experiments share most infrastructure with the Robomimic experiments. To run LIBERO pretraining and finetuning experiments, install the [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) dataset. For pretraining, update the hdf5 and buffer paths in `configs/dataset/libero_90.yaml` and run 
```
python experiments/uwm/train_robomimic.py --config-name train_uwm_robomimic.yaml dataset=libero_90 exp_id=pretrain
```
This will pretrain a UWM on the LIBERO-90 dataset. To finetune this model on a downstream task (e.g., Book-Caddy), update the hdf5 and buffer paths in `configs/dataset/libero_book_caddy.yaml` and run
```
python experiments/uwm/train_robomimic.py --config-name finetune_uwm_robomimic.yaml dataset=libero_book_caddy exp_id=finetune pretrain_checkpoint_path="logdir/uwm/libero_90/pretrain/0/models.pt"
```

## DROID Pretraining / Cotraining / Finetuning Experiments
We provide shell scripts for DROID pretraining / cotraining / finetuning experiments in the `scripts` directory. Each script runs a dataset conversion pipeline to create a Zarr buffer for the corresponding DROID TFDS dataset and then train a model. 

To launch a DROID pretraining experiment, install the [DROID](https://droid-dataset.github.io/) dataset, update the `DATA_DIR` and `BUFFER_PATH` in `scripts/launch_droid_pretrain.sh`, and run the script 
```
source scripts/launch_droid_pretrain.sh
```
To launch a video cotraining experiment, modify and run
```
source scripts/launch_droid_cotrain.sh
```
To fineune a pretrained model to a downstream task, collect demonstrations using the DROID interface and convert then into a TFDS dataset. Then modify and run 
```
source scripts/launch_droid_finetune.sh
```

We release the pretrained and cotrained DROID UWM checkpoints [here](https://drive.google.com/drive/folders/1M4AuVLMRpSwOf_YAp56bV9AqyZI9ul6g?usp=sharing). You can download and directly finetune from these checkpoints.

## Bibtex
If you find this code useful, please cite:

```
@inproceedings{zhu2025uwm,
    author    = {Zhu, Chuning and Yu, Raymond and Feng, Siyuan and Burchfiel, Benjamin and Shah, Paarth and Gupta, Abhishek},
    title     = {Unified World Models: Coupling Video and Action Diffusion for Pretraining on Large Robotic Datasets},
    booktitle = {Proceedings of Robotics: Science and Systems (RSS)},
    year      = {2025},
}
```