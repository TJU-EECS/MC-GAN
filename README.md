# Disclaimer
This is a repository of Markov Conditional Generative Adversarial Network (MCGAN) !!!

# MCGAN — Official PyTorch Implementation
**Title: "A Markov Conditional Generative Adversarial Network for Old Photo Restoration"**  
The code for this paper is based on the Basicsr library. 

# Network architecture:
![network.png](https://github.com/TJU-WEIHAO/MLCN/blob/main/network.png)


# Installation
* Python 3.6 is used. Basic requirements are listed in the 'requirements.txt'.

```
pip install -r requirements.txt
```

* Install the [BasicSR](https://github.com/XPixelGroup/BasicSR) following the instructions.


# Pre-trained networks
We provide pre-trained networks trained on the old photo dataset created by us. Please download *.pth from the [CHECKPOINTS Google Drive folder](https://drive.google.com/drive/folders/1-CWgyodbc_kB0YCPIw89BSS6Oap6UtLc?usp=sharing) and put the downloaded files in ./experiments. After that, please change the options of test_MCGAN.yml.
 
# Testing
We provide processed testing set. Please down the datasets.zip files from the [DATASET Google Drive folder](https://drive.google.com/file/d/1-HJNnFkLEjpXQs4s2BuxNPVPT-X6nwHr/view?usp=sharing) and unzip this files. You should put the unzipped data into ./datasets/. 
Of course, You can use personal data with a similar structure to the test dataset. After that, please change the options of test_MCGAN.yml.

If you want to evaluate the quantitative metrics, run:  
> PYTHONPATH="./:${PYTHONPATH}" CUDA_VISIBLE_DEVICES=0 python basicsr/test.py -opt options/MyOptions/test/test_INPLGAN.yml


The test results will be updata in the ./experiments files.  

# Experiments:
![results.jpg](https://github.com/TJU-WEIHAO/MLCN/blob/main/results.jpg)


``` latex
@misc{wang2020basicsr,
  author =       {Xintao Wang and Ke Yu and Kelvin C.K. Chan and
                  Chao Dong and Chen Change Loy},
  title =        {BasicSR},
  howpublished = {\url{https://github.com/xinntao/BasicSR}},
  year =         {2020}
}
```
