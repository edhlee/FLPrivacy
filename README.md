# FLPrivacy

<p align="center">
  <img src="https://github.com/edhlee/FLPrivacy/blob/main/logos/logo_Nov13.png" width="25%" height="auto"  alt="Hospital Network Under Attack"/>
</p>


Update - Nov 10 (PST). 
Compiled release of privacy experiments - Nov. 10.

Exploring Privacy on a Federated Medical Imaging Platform

## Description

Federated learning (FL) is a distributed approach to training models, enabling model training without exposing raw patient data. However, FL alone does not necessarily imply that privacy is completely preserved. In this work, we explore the extent to which attackers, who have partial knowledge about the FL-dataset or trained models, can infer properties about the data itself using inference-based attacks. We curate one of the most diverse and largest CT dataset to quantify privacy leakage across FL techniques. Using this dataset, we illustrate that FL can provide some baseline level of protection over centralized data sharing (CDS). However, even with FL and Differentially-Private training (DP) on FL, site and membership information are still vulnerable to inference-based attacks. For example, an attacker can achieve  40\% accuracy on identifying the site of origin with knowledge of the trained model and 10\% of the exposed training data. One method that we propose to defend against this in the context of FL is the use of synthetic data augmentation. 

This repository shows the steps to reproduce these attacks and defenses. More updates to come:

## Todo Checklist
- [x] <span style="color: green;">Site, age, sex inference attacks with CDS+FL+GAN trained models</span>
- [ ] GAN code to generate the site-specific synthetic 3D chest CT dataset 
- [ ] DPSGD Code on FL training

## Instructions
To use this project, 


<p align="center">
  <img src="https://github.com/edhlee/FLPrivacy/blob/main/logos/privacy_attack.png" width="75%" height="auto"  alt="21 site attack"/>
</p>


### Prerequisites

See the version of tensorflow used in the requirements.txt. Or set up an environment and run the following.

```bash
pip install -r requirements.txt

