# Disclaimer
This is a repository of Markov Conditional Generative Adversarial Network (MCGAN) !!!

# MCGAN — Official PyTorch Implementation
**Title: "A Markov Conditional Generative Adversarial Network for Old Photo Restoration"**  
The code for this paper is based on the Basicsr library. 

# Network architecture:
![network.png](https://github.com/TJU-EECS/MC-GAN/blob/main/assert/network.jpg)


# Installation

* Install the [BasicSR](https://github.com/XPixelGroup/BasicSR) following the instructions.

* Python 3.6 is used. Basic requirements are listed in the 'requirements.txt'.

```
pip install -r requirements.txt
```

# Pre-trained networks
We provide pre-trained networks trained on the old photo dataset created by us. Please download *.pth from the [checpoints](https://drive.google.com/drive/u/0/folders/1mWnaoIHDgj_sXwByH9sxUAoS3jxCk_uA). Please put the downloaded files in ./experiments and unzip them. After that, please change the options of test_MCGAN.yml.
 
# Testing
We provide processed testing set. Please down the datasets.zip files from the [testsets](https://drive.google.com/drive/u/0/folders/1mWnaoIHDgj_sXwByH9sxUAoS3jxCk_uA) and unzip this files. You should put the unzipped data into ./datasets/. 
Of course, You can use personal data with a similar structure to the test dataset. After that, please change the options of test_MCGAN.yml.

If you want to evaluate the quantitative metrics, run:  
> PYTHONPATH="./:${PYTHONPATH}" CUDA_VISIBLE_DEVICES=0,1 python basicsr/test.py -opt options/MyOptions/test/test_MCGAN.yml

The test results will be updata in the ./experiments files.  

The more details can be find in ./docs.

- **Training and testing commands**: Please see **[TrainTest.md](docs/TrainTest.md)** for the basic usage.
- **Options/Configs**: Please refer to [Config.md](docs/Config.md).
- **Logging**: Please refer to [Logging.md](docs/Logging.md).

# Experiments:
The results of old photo restoration are as follows.
![results.jpg](https://github.com/TJU-EECS/MC-GAN/blob/main/assert/result.jpg)

The results of image denoising and deblurring are as follows.
![results.jpg](https://github.com/TJU-EECS/MC-GAN/blob/main/assert/result1.jpg)

``` latex
@misc{wang2020basicsr,
  author =       {Xintao Wang and Ke Yu and Kelvin C.K. Chan and
                  Chao Dong and Chen Change Loy},
  title =        {BasicSR},
  howpublished = {\url{https://github.com/xinntao/BasicSR}},
  year =         {2020}
}
```
