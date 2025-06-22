# DLproject
A project about data poisoning on satellite images. 

# Get the data
Data is not directly available here as it's too big for github. However you can download it [here](https://zenodo.org/records/7711810#.ZAm3k-zMKEA):
```bash
wget https://zenodo.org/records/7711810/files/EuroSAT_RGB.zip?download=1
```
# How to run
In order to train the models, just use the `train.py` script:
```python
python3 train.py --models cnn resnet googlenet
```
ViT (vision transformer) is listed as an option but not currently working yet.

The training parameters are all contained in `config.py` and should only be modified there. 

# Data poisoning part
Check [here](https://github.com/GiovanniBillo/data-poisoning#) for now: i will integrate the repos later.
Run the data poisoning experiment with:
```python
python3 brew_poison.py --dataset EUROSAT 
```
It could happen that the program complains about Local Cache (expecially if running on google colab). In that case, reinstall the dataset library: `pip install -U datasets`.
Running this complete experiment takes approximately 2 hours.

## References
Dataset:
[EUROSAT](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8736785)
Types of attacks
- [Gradient Matching](https://openreview.net/pdf?id=01olnfLIbD)
- [Bullseye Polytope](https://arxiv.org/pdf/2005.00191)
Architectures:
- [ViT](https://arxiv.org/pdf/2010.11929) 
