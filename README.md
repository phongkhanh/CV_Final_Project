# Final Project Of Computer Vision

### Pipeline of our tracking method
![image](image/overall.png)


### Install
This implementation uses Python 3.8, [Pytorch](http://pytorch.org/),  Cuda 11.3. 
```shell
# Copy/Paste the snippet in a terminal
git clone https://github.com/tinery/AICUP-2024-competition.git
cd AICUP-2024-competition

#Dependencies
conda create -n Tracker python=3.8 --yes
conda activate Tracker
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip install --user --requirement  requirements.txt # pip dependencies
```

### Weights
Link drive [Weights](https://drive.google.com/drive/folders/1mLgClpvm73F2PfR_laCfPdZw_sNB_09q?usp=sharing)

### Usage

Demo :    ```./run_submit.sh``` <br>
Demo with colab: [ipynb file](https://colab.research.google.com/drive/1xrGFTbhR0yjhzXqRxamLNtGMQAe-V5sH#scrollTo=3eR65IfWoKbz)
