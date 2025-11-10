# Following the Trace: Radiologists’ Visual Cognition-guided Cooperative Network for Medical Image Diagnosis

This project contains the training and testing code for the paper, as well as the model weights trained according to our algorithm

> Computer-aided diagnosis (CAD) has significantly advanced automated chest X-ray analysis but remains an isolated computational tool, detached from clinical workflows and lacking reliable decision support and interpretability. Human AI collaboration seeks to enhance the reliability of diagnostic models by integrating controllable human priors. However, the semantic gap between radiologists’ prior knowledge and model representations usually impedes collaboration, while the absence of interactive tools seamlessly embedded within diagnostic routines further limits clinical adoption.
To overcome these limitations, we propose a Visual Cognition-guided Collaborative Network (VCC-Net). VCC-Net centers on visual cognition (VC) and employs cost-effective, clinically compatible interfaces, such as eye-tracking or mouse interactions, to capture radiologists’ visual search traces and attention patterns during diagnosis and convert them into structured signals.
VCC-Net employs VC as a spatial perceptual guide, learning hierarchical visual search strategies to localize diagnostically key regions. A cognition–graph co-editing mechanism subsequently integrates human priors with model inference to construct a pathology-aware graph. The mechanism captures dependencies among anatomical regions and aligns model representations with VC-driven features, mitigating human bias and facilitating complementary, transparent decision-making.
Experiments on the public datasets SIIM-ACR, EGD-CXR, and self-constructed TB-Mouse dataset achieved classification accuracies of 88.40\%, 85.05\%, and 92.41\%, respectively. 
The attention maps produced by VCC-Net exhibit strong concordance with radiologists’ gaze distributions. The findings demonstrate a mutual reinforcement of human expertise and model inference, advancing CAD from a conventional assistive tool toward a new paradigm of collaborative diagnostic intelligence.

## Methods
![](./Fig/method.png)

## Qualitative Results
![1.0](Fig/result1.png)
![1.0](Fig/result2.png)


## Model Weights
The download links and extraction codes for our model weights are as follows：
[Checkpoint](https://pan.baidu.com/s/1CxQ5CIH4ol3hBt_-Lk5ksg?pwd=7777) . 
<br>
Code: 7777 

## Dataset
* The SIIM-ACR dataset can be downloaded from  [SIIM-ACR](https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation/data) .
* The gaze data for SIIM-ACR can be downloaded at [Gaze of SIIM-ACR](https://github.com/HazyResearch/observational) .
* The EGD-CXR dataset can be downloaded from [EGD-CXR](https://physionet.org/content/egd-cxr/1.0.0/) .

## Requirements
```
python == 3.8
torch == 1.12.0
numpy == 1.24.4
medpy == 0.5.1
nibabel == 5.2.1
pandas == 2.0.3
scikit-image == 0.21.0
```
