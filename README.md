### Overview

This repository provides an algorithm that performs gradient updates on the perturbed weights of a graph neural network, named noise stability optimization. Besides the implementation of our algorithm, we also provide the implementation to evaluate Hessian-based quantities (e.g., traces, top-eigenvalues, Hessian vector product) of fine-tuned GNNs. Our observation is that the Hessian-based measurements correlate better with observed generalization gaps of fine-tuned GNNs. 

### Requirements

We use the Python packages in `requirements.txt` for development. To install requirements:

```
pip install -r requirements.txt
```

### Dataset and Pretrained models

- Dataset: Please download the dataset from the [link](http://snap.stanford.edu/gnn-pretrain/data/chem_dataset.zip) and unzip it under the `./src` folder. 
- Pretrained models: Please download the pretrained model from the [link](https://github.com/snap-stanford/pretrain-gnns/tree/master/chem/model_gin) and put them in the `./src/model_gin/` folder. 

Our code is built on the project of ["Strategies for Pre-training Graph Neural Networks" from Hu et al](https://github.com/snap-stanford/pretrain-gnns). Thanks to the authors for providing their implementation and dataset online. 

### Usage

##### Fine-tuning pretrained GNNs on molecular prediction datasets

Use `finetune.py` to run experiments of fine-tuning on pretrained GNN models. Choose the dataset from sider, clintox, bace, bbbp, and tox21. Use `--nsm_sigma`, `--nsm_lam`, and `--num_perturbs` to change the hyper-parameters: sigma, lambda, and number of perturbations. We search the sigma in $[0.02, 0.05, 0.1]$ and the lambda in $[0.4, 0.6, 0.8]$.  The following bash script shows an example to run the commands. 

```bash
python finetune.py --input_model_file model_gin/supervised_masking.pth --split scaffold --gnn_type gin --dataset sider --device 0\
--train_nsm --nsm_sigma 0.1 --nsm_lam 0.6 --use_neg --reg_method penalty --lam_gnn 1e-4 --lam_pred 1e-4

python finetune.py --input_model_file model_gin/supervised_masking.pth --split scaffold --gnn_type gin --dataset clintox --device 0\
--train_nsm --nsm_sigma 0.05 --nsm_lam 0.4 --use_neg --reg_method penalty --lam_gnn 1e-4 --lam_pred 1e-4

python finetune.py --input_model_file model_gin/supervised_contextpred.pth --split scaffold --gnn_type gin --dataset bace --device 0\
--train_nsm --nsm_sigma 0.1 --nsm_lam 0.6 --use_neg --reg_method penalty --lam_gnn 1e-4 --lam_pred 1e-4

python finetune.py --input_model_file model_gin/supervised_contextpred.pth --split scaffold --gnn_type gin --dataset bbbp --device 0\
--train_nsm --nsm_sigma 0.05 --nsm_lam 0.6 --use_neg --reg_method penalty --lam_gnn 1e-4 --lam_pred 1e-4

python finetune.py --input_model_file model_gin/supervised_edgepred.pth --split scaffold --gnn_type gin --dataset tox21 --device 0\
--train_nsm --nsm_sigma 0.1 --nsm_lam 0.4 --use_neg --reg_method penalty --lam_gnn 1e-4 --lam_pred 1e-4
```

##### Evaluating the Hessian-based measures

Use the following scripts to compute Hessian-based measurements. We use Hessian vector multiplication tools from PyHessian (Yao et al., 2020).

- `compute_hessian_spectra.py` computes the trace and the eigenvalues of the loss's Hessian matrix of each layer in a neural network.
- `compute_hessian_norms.py` computes the Hessian-based vector product.

Please follow the bash script examples to run the commands. Specify the `checkpoint_name` and `dataset` for computing the quantities. 

```bash
python compute_hessian_spectra.py --input_model_file model_gin/supervised_contextpred.pth --split scaffold --gnn_type gin --dataset $dataset --batch_size 32 --device 0 --checkpoint_name $checkpoint_name

python compute_hessian_trace.py --input_model_file model_gin/supervised_contextpred.pth --split scaffold --gnn_type gin --dataset $dataset --batch_size 32 --device 0 --checkpoint_name $checkpoint_name
```

### Citation

If you find this repository useful or happen to use it in a research paper, please cite our work with the following bib information.

```tex
@article{ju2023generalization,
  title={Generalization in Graph Neural Networks: Improved PAC-Bayesian Bounds on Graph Diffusion},
  author={Ju, Haotian and Li, Dongyue and Sharma, Aneesh and Zhang, Hongyang R},
  journal={International Conference on Artificial Intelligence and Statistics},
  year={2023}
}
```

### Acknowledgment

Thanks to the authors of the following repositories for providing their implementation publicly available, which greatly helps us develop this code. 

- **[Strategies for Pre-training Graph Neural Networks](https://github.com/snap-stanford/pretrain-gnns)**
- **[SAM Optimizer (In PyTorch)](https://github.com/davda54/sam)**
- **[PyHessian](https://github.com/amirgholami/PyHessian)**

