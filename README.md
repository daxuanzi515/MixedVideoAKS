# README

In short, my record and testing tips here, depending on my environment: Ubuntu 20.04, 3090Ti, CUDA 11.3, Python 3.9.

## Requirements Download
### Features Extractor
Just follow the commands:
```bash
conda create -n SeViLA python=3.9
conda activate SeViLA
git clone https://github.com/Yui010206/SeViLA.git
cd SeViLA
pip install -e .
pip install numpy==1.24.4
pip install spacy
```
### MLLMs Deployment
Pay attention to the version of the lmms-eval. The latest version may not work in python 3.9. 
Please use the following commands to install the environment:

```bash 
conda create -n AKS python=3.9
conda activate AKS
git clone https://github.com/LLaVA-VL/LLaVA-NeXT
cd LLaVA-NeXT
pip install -e ".[train]"
cd ..
git clone https://github.com/EvolvingLMMs-Lab/lmms-eval -b v0.2.4
cd lmms-eval
pip install -e .
```

