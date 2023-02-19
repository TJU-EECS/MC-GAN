# Disclaimer
This is a repository of Markov Conditional Generative Adversarial Network (MCGAN) !!!

# MCGAN — Official PyTorch Implementation
**Title: "A Markov Conditional Generative Adversarial Network for Old Photo Restoration"**  
The code for this paper is based on the Basicsr library. 

# Network architecture:
![network.png](https://github.com/TJU-WEIHAO/MLCN/blob/main/network.png)


# Installation
Clone this repository:  
> git clone https://github.com/TJU-WEIHAO/MLCN.git  
> cd ./MLCN/

Create a virtual environment:
> conda create -n [name] python=3.8  
> conda activate [name]

Install PyTorch and other dependencies:  
> pytorch == 1.8  
> torchvision == 0.9.0  
> opencv == 4.5.3  
> scipy == 1.7.1  
> pillow == 8.3.1 

Install deformable convolution:  
> sh make.sh  

# Pre-trained networks
We provide pre-trained networks trained on the VITON dataset. Please download *.pth from the [CHECKPOINTS Google Drive folder](https://drive.google.com/drive/folders/1-CWgyodbc_kB0YCPIw89BSS6Oap6UtLc?usp=sharing) and put the downloaded files in ./checkpoints/ directory.
 
# Testing
We provide processed testing set. Please down the datasets.zip files from the [DATASET Google Drive folder](https://drive.google.com/file/d/1-HJNnFkLEjpXQs4s2BuxNPVPT-X6nwHr/view?usp=sharing) and unzip this files. test.py assumes that the downloaded files are placed in ./datasets/ directory.  
If you want to evaluate the quantitative metrics, run:  
> python test.py --list paired_list.txt  

If you want to obtain the qualitative results, run:  
> python test.py --list unpaired_list.txt  

The test results will be updata in the test_results files.  

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
