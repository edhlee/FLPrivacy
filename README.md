# FLPrivacy

<p align="center">
  <img src="https://github.com/edhlee/FLPrivacy/blob/main/logos/logo_Nov13.png" width="25%" height="auto"  alt="Hospital Network Under Attack"/>
</p>


Update - Nov 10 (PST). 
Compiled release of privacy experiments - Nov. 10.

**Exploring Privacy on a Federated Medical Imaging Platform**

## Description

Federated learning (FL) is a distributed approach to training models, enabling model training without exposing raw patient data. However, FL alone does not necessarily imply that privacy is completely preserved. In this work, we explore the extent to which attackers, who have partial knowledge about the FL-dataset or trained models, can infer properties about the data itself using inference-based attacks. We curate one of the most diverse and largest CT dataset to quantify privacy leakage across FL techniques. Using this dataset, we illustrate that FL can provide some baseline level of protection over centralized data sharing (CDS). However, even with FL and Differentially-Private training (DP) on FL, site and membership information are still vulnerable to inference-based attacks. For example, an attacker can achieve  40\% accuracy on identifying the site of origin with knowledge of the trained model and 10\% of the exposed training data. One method that we propose to defend against this in the context of FL is the use of synthetic data augmentation. 

This repository shows the steps to reproduce these attacks and defenses. There will be updates such as the code base to generate (conditionally) the synthetic medical images that are site-specific, the DPSGD code and the release of de-identified data with permission from the site contributors. 

## Todo Checklist
- [x] <span style="color: green;">Site, age, sex inference attacks with CDS+FL+GAN trained models</span>
- [x] <span style="color: green;">Visualizations to illustrate the effects of all attacks</span>
- [ ] GAN code to generate the site-specific synthetic 3D chest CT dataset 
- [ ] DPSGD Code on FL training

## Instructions
To perform a site inference attack, please see the directory fed_privacy_attack/. First install the prereqs (see below) and download the necessary data from Drive (note that this is temporary as we plan to move to AWS). If one needs help with data, please contact edhlee@stanford.edu or edward.heesung.lee@gmail.com. Next, we will run inference attacks on different trained model files. Please access the [Data](https://drive.google.com/file/d/1fySrSXNZAkPedDmncTAtuhqm52arroiV/view?usp=drive_link), and unzip in data/.
<p align="center">
  <img src="https://github.com/edhlee/FLPrivacy/blob/main/logos/privacy_attack.png" width="75%" height="auto"  alt="21 site attack"/>
</p>


Please see the version of tensorflow used in the requirements.txt. Or set up an environment and run the following. The typical install time is a few minutes. The typical run time on a Volta or higher GPU is 10 minutes for ./sample_train_attack.sh on 4 Volta GPUs (10% exposed training data per site). In the screenshot below, you will see the attack results after the adversary trains on the CDS model, Fed learning model trained with DPSGD, and a randomly initialized model.


 <p align="center">
   <img src="https://github.com/edhlee/FLPrivacy/blob/main/logos/terminal_screenshot_privacy.png" width="75%" height="auto"  alt="21 site attack"/>
 </p>


```bash
pip install -r requirements.txt

Run 

./sample_train_attack.sh

This will run 3 attacks on 4 trained models (CDS, FL-DPSGD, random)  with 10% leaked data. The output should look similar to the following for randomly initialized:

epoch: 1 model config: 0 f1_weighted: 0.0563582144677639
epoch: 2 model config: 0 f1_weighted: 0.054282404482364655
epoch: 3 model config: 0 f1_weighted: 0.07864472270011902
epoch: 4 model config: 0 f1_weighted: 0.10713225603103638
epoch: 5 model config: 0 f1_weighted: 0.16577060520648956
epoch: 6 model config: 0 f1_weighted: 0.2034904509782791
epoch: 7 model config: 0 f1_weighted: 0.22265328466892242
epoch: 8 model config: 0 f1_weighted: 0.23172716796398163
epoch: 9 model config: 0 f1_weighted: 0.2500138282775879
epoch: 10 model config: 0 f1_weighted: 0.2561897337436676
epoch: 11 model config: 0 f1_weighted: 0.267215371131897
epoch: 12 model config: 0 f1_weighted: 0.2725270688533783
epoch: 13 model config: 0 f1_weighted: 0.26315581798553467
epoch: 14 model config: 0 f1_weighted: 0.2600475549697876
epoch: 15 model config: 0 f1_weighted: 0.26167938113212585
epoch: 16 model config: 0 f1_weighted: 0.2662930190563202
epoch: 17 model config: 0 f1_weighted: 0.26496851444244385
epoch: 18 model config: 0 f1_weighted: 0.25547945499420166
epoch: 19 model config: 0 f1_weighted: 0.2457115650177002
epoch: 20 model config: 0 f1_weighted: 0.24283328652381897


Finally, to visualize the training runs, please see:

visualize_privacy_site_2_confusion.ipynb 






