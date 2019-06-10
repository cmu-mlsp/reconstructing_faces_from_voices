# Reconstructing faces from voices

Implementation of *Reconstructing faces from voices* paper 

[https://arxiv.org/abs/1905.10604](https://arxiv.org/abs/1905.10604)

## Requirements

This implementation is based on Python 3.7 and Pytorch 1.1. 

We recommend you use conda to install the dependencies. All the requirements are found in `requirements.txt`. Run the following command to create a new conda environment using all the dependencies. 

```
$ ./install.sh
```

After you run the above script, you need to activate the environment where all the packages had been installed. The environment is called `voice2face` and can be run by:

```
$ source activate voice2face
```

## Processed data

The following are the **processed** training data we used for this paper. Please feel free to download them. 

Voice data (log mel-spectrograms): [google drive](https://drive.google.com/open?id=1T5Mv_7FC2ZfrjQu17Rn9E24IOgdii4tj)

Face data (aligned face images): [google drive](https://drive.google.com/open?id=1qmxGwW5_lNQbTqwW81yPObJ-S-n3rpXp)

Once downloaded, update variables `voice_dir` and `face_dir` with the corresponding paths. 

### Configurations 

See `config.py` on how to change configurations. 

## Train

```
$ python gan_train.py
```
The trained model is in `models/generator.pth`

## Test

```
$ python gan_test.py
``` 

Results will be in `data/test_data/`

## Citation

	@article{wen2019reconstructing,
	  title={Reconstructing faces from voices},
	  author={Yandong Wen, Rita Singh, Bhiksha Raj},
	  journal={arXiv preprint arXiv:1905.10604},
	  year={2019}
	}

## License 

Check LICENSE.md. 
