# DLproject
A project about data poisoning on satellite images. 

# Get the data
Data is not directly available here as it's too big for github. However you can download it [here](https://zenodo.org/records/7711810#.ZAm3k-zMKEA):
```bash
wget https://zenodo.org/records/7711810/files/EuroSAT_RGB.zip?download=1
```
# How to run
In order to train the models, just use the `train.py` script:
```
python3 train.py --models cnn resnet googlenet
```python
(vit (vision transformer) is listed as an option but not currently working).

The training parameters are all contained in `config.py` and should only be modified there. 
