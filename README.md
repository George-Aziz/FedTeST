# FedTeST 

This repository hosts the supplementary code for my paper titled "Selective Adaptation for Personalised Federated Transformers".
- We propose FedTeST, a personalised federated transformer framework that combines hypernetwork-based attention personalisation with selective train- and test-time adaptation of the transformer output projection, enabling efficient client-specific personalisation under non-IID conditions without full-model adaptation.
## <ins>Recommended</ins> Environment Setup

(1) Clone the repository 
```shell
git clone https://github.com/George-Aziz/FedTeST
```
(2) Install CUDA Toolkit via https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64
* Especially important if you are using WSL 2

(3) Create a new conda environment
```shell
conda create -n <Env_Name> python=3.11 scikit-learn numpy scipy pandas requests einops faiss-gpu tensorboard -c conda-forge
conda activate <Env_Name>
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```
* PyTorch, TorchVision, and TorchAudio are all installed with CUDA 12.8 support, as the experiments in this paper were conducted on an RTX 5090 GPU.
* Running the KNN configuration requires FAISS. For more detailed instructions and installation options, refer to the [FAISS GitHub documentation](https://github.com/facebookresearch/faiss/blob/main/INSTALL.md).

## Datasets

We provide three federated benchmark datasets spanning image classification task (CIFAR10 and CIFAR100) and language modeling(Shakespeare).

* To use the CIFAR10 and CIFAR100 datasets, you may either manually download and extract them into the 'data' directory, or simply run any supported algorithm; the dataset will be downloaded automatically if not already present.
* The Shakespeare dataset was naturally partitioned by assigning all lines from the same character to the same client. 
    * See the `README.md` file in `data/shakespeare` for instructions on generating data before running experiments.

The following table summarizes the datasets and models

|Dataset         | Task |  Model |
| ------------------  |  ------|------- |
| CIFAR10   |     Image classification        |      vit/cnn/cnn-b |
| CIFAR100    |     Image classification         |      vit/cnn/cnn-b  |
| Shakespeare |     Next character prediction        |      transformer/Stacked LSTM    |



## Usage
The following is an example to run our FedTeST :
```
python main.py --model=vit \
 --dataset=cifar10 \ 
 --alg=FedTeST  \
 --lr=0.01 \
 --batch-size=64 \
 --epochs=5 \
 --n_parties=50 \
 --mu=0.01 \
 --rho=0.9 \
 --comm_round=500 \
 --partition=noniid-labeldir \
 --beta=0.3 \
 --device='cuda:0' \
 --datadir='./data/' \
 --logdir='./logs/' \
 --noise=0 \
 --sample=0.1 \
 --init_seed=0 \
 --train_acc_pre \
 --test_round=400 \
 --wandb_flag \
 --train_adapt_method=progressive \
 --activate_refine_round=300  \
 --adapt_target=attn_out
```
All scripts used in the paper are provided in the scrips folder.

| Parameter                      | Description                                 |
| ----------------------------- | ---------------------------------------- |
| `model` | The model architecture. Options: `cnn`, `cnn-b`, `vit`, `lstm`, `transformer`. Default = `vit`. |
| `dataset`      | Dataset to use. Options: `cifar10`, `cifar100`, `shakespeare`. Default = `cifar10`. |
| `alg` | Basic training algorithm. Basic Options: `fedavg`, `fedprox`, `FedTeST `, `FedTP`, `pFedHN`, `pfedMe`, `fedPer`, `fedBN`, `fedRod`, `fedproto`, `local_training`. Extension: `Personalized-T`, `FedTP-Per`, `FedTP-Rod`. Default = `FedTeST `. |
| `lr` | Learning rate for the local models, default = `0.01`. |
| `batch-size` | Batch size, default = `64`. |
| `epochs` | Number of local training epochs, default = `5`. |
| `n_parties` | Number of parties/clients, default = `10`. |
| `mu` | The proximal term parameter for FedProx, default = `1`. |
| `rho` | The parameter controlling the momentum SGD, default = `0`. |
| `comm_round`    | Number of communication rounds to use, default = `500`. |
| `eval_step`    | Test interval during communication, default = `5`. |
| `test_round`    | Round beginning to test, default = `400`. |
| `train_adapt_method` | Method for training when running the adaptive module refinement component, default = `batch`.|
| `adapt_target` | Transformer/ViT Parameter subset to adapt, default = `all`.|
| `activate_refine_round` | Activatation round for the adaptive module refinement component, default = `300`. |
| `train_acc_pre` | Activate global train set evaluations. |
| `partition`    | The partition way. Options: `noniid-labeldir`, `noniid-labeldir100`, `noniid-labeluni`, `iid-label100`, `homo`. Default = `noniid-labeldir`. |
| `beta` | The concentration parameter of the Dirichlet distribution for heterogeneous partition, default = `0.3`. |
| `device` | Specify the device to run the program, default = `cuda:0`. |
| `datadir` | The path of the dataset, default = `./data/`. |
| `logdir` | The path to store the logs, default = `./logs/`. |
| `sample` | Ratio of parties/clients that participate in each communication round, default = `0.1`. |
| `balanced_soft_max` | Activate this to run FedRod and FedTP-Rod. |
| `k_neighbor` | Activate this to run FedTP-KNN. |
| `init_seed` | The initial seed, default = `0`. |
| `noise` | Maximum variance of Gaussian noise we add to local party, default = `0`. |
| `noise_type` | Noise type. Use `increasing` to check effect of heterogeneity in Noise-based Feature Imbalance, default = `None`. |
| `save_model` | Activate this to save model. |
| `wandb_flag` | Activate wandb logging. |



## Data Partition Map
To simulate non-IID scenarios for CIFAR-10/CIFAR-100, we follow two common split designs. You can call function `get_partition_dict()` in `main.py` to access `net_dataidx_map`. `net_dataidx_map` is a dictionary. Its keys are party ID, and the value of each key is a list containing index of data assigned to this party. For our experiments, we usually set `init_seed=0`.  The default value of `noise` is 0 unless stated. We list the way to get our data partition as follow.
* **Dirichlet Partition**: `partition`=`noniid-labeldir/noniid-labeldir100`. The former is for CIFAR-10 dataset and the latter is for CIFAR-100 dataset. `beta` controls degree of data heterogeneity. 
* **Pathological Partition**: `partition`=`noniid-labeluni`. For CIFAR-10 and CIFAR-100 dataset. 


Here is explanation of parameter for function `get_partition_dict()`. 

| Parameter                      | Description                                 |
| ----------------------------- | ---------------------------------------- |
| `dataset`      | Dataset to use. Options: `cifar10`, `cifar100` |
| `partition`    | The partition approach. Options: `noniid-labeldir`, `noniid-labeldir100`, `noniid-labeluni`, `iid-label100`, `homo` |
| `n_parties` | Number of parties. |
| `init_seed` | The initial seed. |
| `datadir` | The path of the dataset. |
| `logdir` | The path to store the logs. |
| `beta` | The concentration parameter of the Dirichlet distribution for heterogeneous partition. |

## Acknowledgements

(1) Baseline solution from [FedTP](https://github.com/zhyczy/FedTP)

(2) Adaptive Module Refinement Component inspired from [ATP](https://github.com/baowenxuan/ATP/tree/master)

(3) Datasets used in the paper
* [CIFAR-10](https://docs.pytorch.org/vision/main/generated/torchvision.datasets.CIFAR10.html)
* [CIFAR-100](https://docs.pytorch.org/vision/main/generated/torchvision.datasets.CIFAR100.html)
* [Shakespeare](https://github.com/TalwalkarLab/leaf/tree/master/data/shakespeare)
