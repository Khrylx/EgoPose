# EgoPose
![Loading EgoPose demo gif](https://www.ye-yuan.com/wp-content/uploads/2019/08/ego_pose_rs.gif "EgoPose demo gif")
![Loading EgoPose demo gif](https://www.ye-yuan.com/wp-content/uploads/2019/07/ego_forecast_rs.gif "EgoPose demo gif")
---
This repo contains the official implementation of our paper:
  
Ego-Pose Estimation and Forecasting as Real-Time PD Control  
Ye Yuan, Kris Kitani  
**ICCV 2019**  
[[website](https://www.ye-yuan.com/ego-pose)] [[paper](https://arxiv.org/pdf/1906.03173.pdf)] [[video](https://youtu.be/968IIDZeWE0)]

# Installation 
### Dataset
* Download the dataset from Google Drive in the form of a [single zip](https://drive.google.com/file/d/1vzxVHAtfvfIEDreqYvHulhtNwHcomotV/view?usp=sharing) or [split zips](https://drive.google.com/drive/folders/1gArkvMsyePQvSkWaV45708MchH1nDZZE?usp=sharing) (or [BaiduYun link](https://pan.baidu.com/s/18iSI84nFpCUdAqhuN1PcWw), password: ynui) and place the unzipped dataset folder inside the repo as "EgoPose/datasets". Please see the README.txt inside the folder for details about the dataset.
### Environment
* **Supported OS:** MacOS, Linux
* **Packages:**
    * Python >= 3.6
    * PyTorch >= 0.4 ([https://pytorch.org/](https://pytorch.org/))
    * gym ([https://github.com/openai/gym](https://github.com/openai/gym))
    * mujoco-py ([https://github.com/openai/mujoco-py](https://github.com/openai/mujoco-py)) (**[MuJoCo](https://mujoco.org/) is now free thanks to DeepMind!** 🎉🎉🎉)
    * OpenCV: ```conda install -c menpo opencv```
    * Tensorflow, OpenGL, yaml: 
    ```conda install tensorflow pyopengl pyyaml```
* **Additional setup:**
    * For linux, the following environment variable needs to be set to greatly improve multi-threaded sampling performance:    
    ```export OMP_NUM_THREADS=1```
* **Note**: All scripts should be run from the root of this repo.

### Pretrained Models
* Download our pretrained models from this [link](https://drive.google.com/file/d/1DE-uSUk4JMDtL9aQY2R5rAd3_yPRUIIH/view?usp=sharing) (or [BaiduYun link](https://pan.baidu.com/s/1NECDEX-itgzKoYHrxSMEwQ), password: kieq) and place the unzipped results folder inside the repo as "EgoPose/results".

# Quick Demo  
### Ego-Pose Estimation
* To visualize the results for MoCap data:  
    ```python ego_pose/eval_pose.py --egomimic-cfg subject_03 --statereg-cfg subject_03 --mode vis```  
    Here we use the config file for subject_03. Note that in the visualization, the red humanoid represents the GT.
    
* To visualize the results for in-the-wild data:  
    ```python ego_pose/eval_pose_wild.py --egomimic-cfg cross_01 --statereg-cfg cross_01 --data wild_01 --mode vis```  
    Here we use the config file for cross-subject model (cross_01) and test it on in-the-wild data (wild_01).
    
* Keyboard shortcuts for the visualizer: [keymap.md](https://github.com/Khrylx/EgoPose/blob/master/docs/keymap.md)
### Ego-Pose Forecasting
* To visualize the results for MoCap data:  
    ```python ego_pose/eval_forecast.py --egoforecast-cfg subject_03 --mode vis```  

* To visualize the results for in-the-wild data:  
    ```python ego_pose/eval_forecast_wild.py --egoforecast-cfg cross_01 --data wild_01 --mode vis```  


# Training and Testing
* If you are interested in training and testing with our code, please see [train_and_test.md](https://github.com/Khrylx/EgoPose/blob/master/docs/train_and_test.md).

# Citation
If you find our work useful in your research, please cite our paper [Ego-Pose Estimation and Forecasting as Real-Time PD Control](https://www.ye-yuan.com/ego-pose):
```
@inproceedings{yuan2019ego,
  title={Ego-Pose Estimation and Forecasting as Real-Time PD Control},
  author={Yuan, Ye and Kitani, Kris},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision (ICCV)},
  year={2019},
  pages={10082--10092}
}
```

# License

The software in this repo is freely available for free non-commercial use. Please see the [license](https://github.com/Khrylx/EgoPose/blob/master/LICENSE) for further details.
