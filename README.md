# Reconstructing faces from voices

Implementation of *Reconstructing faces from voices* paper 

[https://arxiv.org/abs/1905.10604](https://arxiv.org/abs/1905.10604)

## Requirements

This implementation is based on Python 3.7 and Pytorch 1.1. 

We recommend you use conda to install the dependencies. All the requirements are found in `requirements.txt`. Run the following command to create a new conda environment using all the dependencies. 

```
$ conda create --name voice2face --file requirements.txt
```

## Processed data
Voice data (log mel-spectrograms): [google drive](https://drive.google.com/open?id=1T5Mv_7FC2ZfrjQu17Rn9E24IOgdii4tj)
Face data (aligned face images): [google drive](https://drive.google.com/open?id=1qmxGwW5_lNQbTqwW81yPObJ-S-n3rpXp)

### Configurations 

See `config.py` on how to change configurations. 


## Train

```
python gan_train.py
```
The trained model is in the folder of *models/generator.pth*

## Test

```
python gan_test.py
``` 
The results are in the folder of *data/test_data/*
## Citation

	@article{wen2019reconstructing,
	  title={Reconstructing faces from voices},
	  author={Yandong Wen, Rita Singh, Bhiksha Raj},
	  journal={arXiv preprint arXiv:1905.10604},
	  year={2019}
	}
## License 

@TODO: add license
