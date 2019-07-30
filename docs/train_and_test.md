# Training
**Note**: All scripts should be run from the root of this repo.
### Config Files
* Our code uses config files for training state regression  ("config/statereg"), ego-pose estimation  ("config/egomimic"), and ego-pose forecasting ("config/").
Please see .yml files in these folders for parameter settings.
    * The parameter **meta_id** specifies the meta file to be used in training, which defines training and testing data.
    * For ego-pose estimation and state regression, The parameter **fr_margin** specifies the number of padding frames as mentioned in the implementation details of [the paper](https://arxiv.org/pdf/1906.03173.pdf).
      For ego-pose forecasting, **fr_margin** corresponds to the number of past frames.
* When training, each config will generate a folder with the config name under "results/" including the following sub-folders:
    * **models:** saves model checkpoints.
    * **results:** contains saved test results.
    * **log:** contains logs recoreded by the logger.
    * **tb:** contains Tensorboard logs.
    
### Training State Regression
The state regression network provides initial state estimation for our ego-pose estimation, and it is also used in the fail-safe mechanism to reset the humanoid state.
Additionally, it pretrains the ResNet-18 and is used to precompute the CNN features for ego-pose estimation and forecasting. 

* If you have setup your config files "config/statereg/subject_02.yml", run the following command to train the state regression network:  
    ```python ego_pose/state_reg.py --cfg subject_02```  
  And it will save model checkpoints under "results/statereg/subject_02/models/".

### Training Ego-Pose Estimation
1. **Generate CNN features.** we need to use trained state regression network to precompute CNN features. Run the following script:  
    ```python ego_pose/data_process/gen_cnn_feature.py --cfg subject_02 --iter 100 --out-id subject_02```   
    It means using the state regression net from config "subject_02" and iter 100 to generate pickled cnn features "cnn_feat_subject_02.p", which will be saved under "datasets/features/".
    Note that you may find some features already exist, but they are only compatible with predefined configs and pretrained models. So If you change the config or retrain the state regression network, you need to regenerate CNN features.
    
2. **Generate expert features.** Expert means expert demonstrations, a term often used in imitation learning. 
   In our case, expert demonstrations include trajectory features such as joint orientations and velocities. And we need to precompute these expert features from GT MoCap data using this script:  
   ```python ego_pose/data_process/gen_expert.py --meta-id meta_subject_02 --out-id subject_02```   
   It will generate a pickle file "expert_subject_02.p" under "datasets/features/".
   
3. **Config file**. In your config "config/egomimic/subject_02.yml", make sure to set **meta_id** to "meta_subject_02", and **cnn_feat** and **expert** to "subject_02".
   Also, set **state_net_cfg** to "subject_02", which is the statereg config you used to generate CNN features.

4. Start training with the following script:  
   ```python ego_pose/ego_mimic.py --cfg subject_02```  
   You can additionally use "--num-threads" and "--gpu-index" to choose the number of threads for sampling and the GPU you use.
   
###  Training Ego-Pose Forecasting
1. If you haven't generated CNN and expert features for the meta file you want to use, generate them following the same steps for training ego-pose estimation.

2. Setup the config file "config/egoforecast/subject_02.yml".

3. Start training with the following script:   
   ```python ego_pose/ego_forecast.py --cfg subject_02```  
   

# Testing
### Testing State Regression
* Run the following script to generate test results:  
  ```python ego_pose/state_reg.py --cfg subject_02 --iter 100 --mode test```  
  It means using the model from config "subject_02" and iter 100 to generate predicted motion results, which will be saved as "results/statereg/subject_02/results/iter_100_test.p".
  
### Testing Ego-Pose Estimation
* As ego-pose estimation relies on the state regression network and we are testing with precomputed CNN features, we first generate a lightweight version of the state regression net without CNN:  
  ```python ego_pose/state_reg.py --cfg subject_02 --iter 100 --mode save_inf```

* For testing ego-pose estimation with MoCap data, run the following script:  
  ```python ego_pose/ego_mimic_eval.py --cfg subject_02 --iter 3000```  
  You can use "--render" if you want to visualize the test results generation. The results will be saved as "results/egomimic/subject_02/results/iter_3000_test.p".

* For testing with in-the-wild data, we need to use the config "cross_01" which corresponds to the cross-subject model, so it will generalize better. (Of course, you need to train state regression and ego-pose estimation with config "cross_01" first, or use our pretrained models.)  
    * First make sure that you have generated CNN features using the meta file "meta_wild_01.yml":  
    ```python ego_pose/data_process/gen_cnn_feature.py --cfg cross_01 --iter 100 --meta-id meta_wild_01 --out-id wild_01```  
    * Then you can run the testing script:  
    ```python ego_pose/ego_mimic_eval_wild.py --cfg cross_01 --iter 6000 --test-feat wild_01```  
    The results will be saved as "results/egomimic/cross_01/results/iter_6000_wild_01.p".
    
### Testing Ego-Pose Forecasting
* First, make sure you have generated testing results for corresponding ego-pose estimation config (specified in the forecasting config), as forecasting relies on the estimation to provide the initial state.

* For testing ego-pose estimation with MoCap data, run the following script:  
  ```python ego_pose/ego_forecast_eval.py --cfg subject_02 --iter 3000 --mode save```  
  The results will be saved as "results/egoforecast/subject_02/results/iter_3000_test.p".

* For testing with in-the-wild data, we need to use the config "cross_01" which corresponds to the cross-subject model, so it will generalize better. (Of course, you need to train state regression and ego-pose forecasting with config "cross_01" first, or use our pretrained models.)  
    * Same as ego-pose estimation, first make sure that you have generated CNN features using the meta file "meta_wild_01.yml".  
    * Then you can run the testing script:  
    ```python ego_pose/ego_forecast_eval_wild.py --cfg cross_01 --iter 6000 --test-feat wild_01```  
    The results will be saved as "results/egoforecast/cross_01/results/iter_6000_wild_01.p".
    
# Visualizing Results
Please refer to the [quick demo](https://github.com/Khrylx/EgoPose#quick-demo) on how to visualize saved results. 

