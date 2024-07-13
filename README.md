# DISN: Deep Implicit Surface Network for High-quality Single-view 3D Reconstruction  &nbsp;&nbsp;&nbsp; <img src="images/Adobe-Logos.png" width=120px /><img src="images/USC-Logos.png" width=120px />

## (will incorporate latest updates)
* ### We just released a rendered [datasets](https://github.com/Xharlie/ShapenetRender_more_variation) of Shapenet with more view variations that contains RGB, albedo, depth and normal 2D images. Rendering scripts for both v1 and v2 are available.

Please cite our paper
[DISN: Deep Implicit Surface Network for High-quality Single-view 3D Reconstruction (NeurIPS 2019)](https://arxiv.org/abs/1905.10711)

``` 
@incollection{NIPS2019_8340,
title = {DISN: Deep Implicit Surface Network for High-quality Single-view 3D Reconstruction},
author = {Xu, Qiangeng and Wang, Weiyue and Ceylan, Duygu and Mech, Radomir and Neumann, Ulrich},
booktitle = {Advances in Neural Information Processing Systems 32},
editor = {H. Wallach and H. Larochelle and A. Beygelzimer and F. d\textquotesingle Alch\'{e}-Buc and E. Fox and R. Garnett},
pages = {492--502},
year = {2019},
publisher = {Curran Associates, Inc.},
url = {http://papers.nips.cc/paper/8340-disn-deep-implicit-surface-network-for-high-quality-single-view-3d-reconstruction.pdf}
}
``` 
Primary contact: [Qiangeng Xu*](https://xharlie.github.io/)

<img src="images/result.png"  />


## System Requirements
  * ### GPU: 8卡 V100-32G
  * ### tensorflow 1.15
  ```
  conda create -n disn python=3.7 # python 版本必须小于 3.7 才能安装 1.15 版本的 tensorflow
  pip activate disn
  conda install cudatoolkit=10.0 cudnn  # TensorFlow 1.15 通常与 CUDA 10.0 兼容
  pip install protobuf==3.20.*
  pip install tensorflow-gpu==1.15
  python -c "import tensorflow as tf; print(tf.__version__); print(tf.test.is_built_with_cuda()); print(tf.test.is_gpu_available())" # 验证 Tensorflow 是否能检测到 gpus
  conda install -c https://software.repos.intel.com/python/conda/ -c conda-forge mkl
  python check_mkl.py
  pip install trimesh==2.37.20
  pip install opencv-python
  pip install PyMCubes==0.1.2 # 原始版本的 ./demo/demo.py 里面需要使用 Intel Math Kernel Library (MKL) 共享库文件才能生成 .obj 文件，这样很麻烦，所以我改成了用 PyMCubes来生成.obj文件
  pip install pymesh
  pip install joblib
  pip install trimesh
  pip install mkl
  
  ```
Tensorflow 检测到了 GPU

<img width="1164" alt="image" src="https://github.com/user-attachments/assets/d66aa294-b67b-49c3-a046-861bf69295cd">

* 我尝试了 重装 几十种方法，都没有办法 让 numpy 检测到 mkl 包

## Installation SDF_DISN.tar & cam_DISN.tar
### 1. 下载和解压缩 SDF_DISN.tar (shape prediction network 的 checkpoints)
The shape prediction network takes point cloud data as input:
  * Input shape is (1, 212183, 3), representing 212,183 3D points
  * It processes this through several convolutional layers
  * The output is a predicted signed distance function (SDF) for each point
#### (1) 下载 SDF_DISN.tar
  ```
    cd /data/3dPrinter/5_2-DISN-new-master
    mkdir checkpoint
    cd checkpoint
    gdown --id 1PEXVxXflVqWNqinSMC-hFmFdlMyoMZ7i
    ### or baidu yunpan: https://pan.baidu.com/s/1Zujo84JoTcTW5dUl0AvS_w   extraction code: esy9
  ```
#### (2) 解压缩 SDF_DISN.tar
  ```
    tar -xvzf SDF_DISN.tar
    rm -rf SDF_DISN.tar
  ```
<img width="362" alt="image" src="https://github.com/user-attachments/assets/8a6d8789-0ae1-4708-b4fc-34d0967a4569">

### 2. 下载和解压缩 cam_DISN.tar (camera estimation network 的 checkpoints)
The camera estimation network 预测 a transformation matrix for the input.
#### (1) 下载 cam_DISN.tar
  ```
    cd /data/3dPrinter/5_2-DISN-new-master
    mkdir cam_est/checkpoint
    cd cam_est/checkpoint
    gdown https://drive.google.com/uc?id=1S5Gh_u1C9vDvksqXDn3CP6IqsnU0hKkj
    ### or baidu yunpan: https://pan.baidu.com/s/1lEHmSHA1o5lrswp0TM50qA   extraction code: gbb3
  ```
#### (2) 解压缩 cam_DISN.tar

 ```
    tar -xvzf cam_DISN.tar
    rm -rf cam_DISN.tar
    cd ../../
    Install whatever libary(e.g. mkl) you don't have and change corresponding libary path in your system in isosurface/LIB_PATH
  ```
<img width="359" alt="image" src="https://github.com/user-attachments/assets/37ffc247-73ba-40de-80b5-cb91bbfcded9">

## Demo:
 * --sdf_res control the resolution of the sampled sdf, default is 64, the larger, the more fine-grained, but slower.
  ```
    cd /data/3dPrinter/5_2-DISN-new-master
    source isosurface/LIB_PATH
    chmod +x ./isosurface/computeMarchingCubes
    nohup python -u demo/demo.py --cam_est --log_dir checkpoint/SDF_DISN --cam_log_dir cam_est/checkpoint/cam_DISN --img_feat_twostream --sdf_res 256 &> log/create_sdf.log &
    python -u demo/demo.py --cam_est --log_dir checkpoint/SDF_DISN --cam_log_dir cam_est/checkpoint/cam_DISN --img_feat_twostream --sdf_res 256
  ``` 
  The result is demo/result.obj.
  if you have dependency problems such as your mkl lib, etc. Please install the corresponding dependencies and change the path in LIB_PATH. Everyone has his/her/their own environment setting so it's impossible to instruct this step without sitting besides you and your server.
<img width="1242" alt="image" src="https://github.com/user-attachments/assets/ac3e513c-a143-4038-910f-8a09c1a848fd">


从输出来看，程序似乎运行成功了。让我为您解释一下主要步骤和输出：

1. 程序成功加载了相机估计模型和主要的SDF (Signed Distance Function) 预测模型。

2. GPU被正确识别和使用：
   ```
   Found device 0 with properties: 
   name: Tesla V100S-PCIE-32GB major: 7 minor: 0 memoryClockRate(GHz): 1.597
   pciBusID: 0000:00:06.0
   ```

3. 相机估计网络成功运行并输出了预测的变换矩阵：
   ```
   pred_trans_mat_val [[[-66.2053       3.0475838   -0.37013683]
     [-15.545383   -81.84504     -0.22653787]
     [-45.6831      -1.9617836    0.2427571 ]
     [100.04551     99.63472      1.4153073 ]]]
   ```

4. 主要的SDF预测模型成功加载和运行。它处理了形状为 (80, 1, 212183, 3) 的输入点云数据。

5. 最后，程序开始创建对象并且保存到了 `demo/result.obj`：
   ```
   submit create_obj
   result.obj has been created into demo/result.obj
   ```
<img width="1236" alt="image" src="https://github.com/user-attachments/assets/e19e4871-8f02-4b4e-abd4-c784dfb74575">


 
## Data Preparation

### 1. 下载 ShapeNetCore.v1 数据集
  download the dataset following the instruction of https://www.shapenet.org/account/  (about 30GB)

  直接百度云 下载 ShapeNetCore.v1.zip 
    ```
    链接：https://pan.baidu.com/s/1WnJIAk4slq99GzE08dELqA 
    提取码：aic6 
    ```
    然后解压缩 ShapeNetCore.v1.zip 
  ```
  cd {your download dir}
  unzip ShapeNetCore.v1.zip -d {your mesh_dir}
  ```
  我下载到了 `/data/3dPrinter/0_Dataset_Ori/4_DISN_Datasets/1_Downloaded/ShepeNet/ShapeNetCore_v1`

### 2. 设置 之前下载的 ShapeNetCore_v1 数据集的位置 & 要生成的文件的位置
  * under preprocessing/info.json, you can change the locations of your data: the neccessary dir for the main model are : 
   ```  
   {
       "lst_dir": "./data/filelists", 
       "cats": {
           "watercraft": "04530566",
           "rifle": "04090263",
           "display": "03211117",
           "lamp": "03636649",
           "speaker": "03691459",
           "cabinet": "02933112",
           "chair": "03001627",
           "bench": "02828884",
           "car": "02958343",
           "airplane": "02691156",
           "sofa": "04256520",
           "table": "04379243",
           "phone": "04401088"
       },
       "all_cats": [
           "airplane",
           "bench",
           "cabinet",
           "car",
           "chair",
           "display",
           "lamp",
           "speaker",
           "rifle",
           "sofa",
           "table",
           "phone",
           "watercraft"
       ],
       "raw_dirs_v1": {
           "mesh_dir": "/data/3dPrinter/0_Dataset_Ori/4_DISN_Datasets/1_Downloaded/ShepeNet/ShapeNetCore_v1", # 改成 (1) 里面下载的 ShapeNetCore_v1 的位置， 下面的其他地址都是 后面运行代码生成的东西所需要放的位置，可以随便写
           "norm_mesh_dir": "/data/3dPrinter/0_Dataset_Ori/4_DISN_Datasets/1_Downloaded/ShepeNet/march_cube_objs_v1/",
           "norm_mesh_dir_v2": "/data/3dPrinter/0_Dataset_Ori/4_DISN_Datasets/1_Downloaded/ShepeNet/march_cube_objs/",
           "sdf_dir": "/data/3dPrinter/0_Dataset_Ori/4_DISN_Datasets/1_Downloaded/ShepeNet/SDF_v1/",
           "sdf_dir_v2": "/data/3dPrinter/0_Dataset_Ori/4_DISN_Datasets/1_Downloaded/ShepeNet/SDF_v2/",
           "rendered_dir": "/data/3dPrinter/0_Dataset_Ori/4_DISN_Datasets/1_Downloaded/ShepeNet/ShapeNetRendering/",
           "rendered_new_dir": "/data/3dPrinter/0_Dataset_Ori/4_DISN_Datasets/1_Downloaded/ShepeNet/ShapeNetRendering.v1/",
           "renderedh5_new_dir_easy": "/data/3dPrinter/0_Dataset_Ori/4_DISN_Datasets/1_Downloaded/ShepeNet/ShapeNetRenderingh5_v1_new_easy.v1/",
           "renderedh5_new_dir_hard": "/data/3dPrinter/0_Dataset_Ori/4_DISN_Datasets/1_Downloaded/ShepeNet/ShapeNetRenderingh5_v1_new_hard.v1/",
           "renderedh5_dir": "/data/3dPrinter/0_Dataset_Ori/4_DISN_Datasets/1_Downloaded/ShepeNet/ShapeNetRenderingh5_v1_more/",
           "renderedh5_dir_v2": "/data/3dPrinter/0_Dataset_Ori/4_DISN_Datasets/1_Downloaded/ShepeNet/ShapeNetRenderingh5_v2/",
           "renderedh5_dir_est": "/data/3dPrinter/0_Dataset_Ori/4_DISN_Datasets/1_Downloaded/ShepeNet/ShapeNetRenderingh5_v1_pred_3d/",
           "3dnnsdf_dir": "/data/3dPrinter/0_Dataset_Ori/4_DISN_Datasets/1_Downloaded/ShepeNet/SDF_full/"
       }
   }

   ```

  
### 3. 用 3d.obj 模型 生成 SDF ground truth & 用 marching cube 重建出来的 reconstructed ground truth models 
#### 方法1: 直接下载作者生成好的 sdf tar.gz
  1. 下载 generated sdf tar.gz from [here](https://drive.google.com/file/d/1cHDickPLKLz3smQNpOGXD2W5mkXcy1nq/view?usp=sharing)
  2. then place it at your "sdf_dir" in json;
  3. 下载 the marching cube 重建出来的 reconstructed ground truth models from the sdf file from [here](https://drive.google.com/drive/folders/1QGhDW335L7ra31uw5U-0V7hB-viA0JXr?usp=sharing)
  4. then place it at your "norm_mesh_dir" in your json.
#### 方法2: 用代码生成 SDF 文件 & 用代码生成 the reconstructed models (大概会运行几小时) (这个环境超级难配)
  This step used this paper [Vega: non-linear fem deformable object simulator](http://run.usc.edu/vega/SinSchroederBarbic2012.pdf). 
  因为这个环境超级难配，为了 把 preprocessing/create_point_sdf_grid.py 跑起来，首先需要下载 MKL、tcmalloc、TBB、libpng12 库。并且把 他们的位置放到 preprocessing/create_point_sdf_grid.py 的开头, 如何解决我都已经更新到 preprocessing/create_point_sdf_grid.py 代码里面了，这里简要说一下如何解决的。
#####  1. 下载 libpng12-0
     * 到 https://launchpad.net/~linuxuprising/+archive/ubuntu/libpng12 下载 .deb
     * 访问 PPA 的包详情页面：https://launchpad.net/~linuxuprising/+archive/ubuntu/libpng12/+packages
     * 找到 libpng - 1.2.54-1ubuntu1.1+1~ppa0~focal    (changes file)    logix2    2019-12-07    Published    Focal    Libs    All builds were built successfully.
     * 找到 libpng12-0_1.2.54-1ubuntu1.1+1~ppa0~focal_amd64.deb (168.9 KiB)
     * 下载到 本地的一个目录下
     * 安装 使用 `sudo dpkg -i libpng12-0_1.2.54-1ubuntu1.1+1~ppa0~focal_amd64.deb` 进行安装
#####  2. 下载 TBB
  ```
  # 更新 TBB 的路径
  (disn) (base) root@v100s02:/data/3dPrinter/5_2-DISN-new-master# sudo find / -name "libtbb*.so*" # => 之后会找到一堆路径，然后选择一个
  sudo ln -s /data/3dPrinter/5_2-DISN-new-master/isosurface/tbb/tbb2018_20180822oss/lib/intel64/gcc4.7/libtbb_preview.so.2 /usr/lib/libtbb_preview.so.2 # 创建一个符号链接到 /usr/lib/ 目录
  ```
  之后把上面的 这个 `/data/3dPrinter/5_2-DISN-new-master/isosurface/tbb/tbb2018_20180822oss/lib/intel64/gcc4.7/libtbb_preview.so.2 /usr/lib/libtbb_preview.so.2` 地址更新到 preprocessing/create_point_sdf_grid.py 的开头，更新到环境变量里面。
#####  3. 对于 MKL, tcmalloc 也一样，按照 TBB 的步骤就可以了
#####  4. 运行 preprocessing/create_point_sdf_grid.py 来生成 ShapeNetCore_V1 文件夹下面各种 .obj 所对应的 .sdf 文件 & 用 marching cube 重建出来的 reconstructed ground truth models 
  ```
  mkdir log
  cd /data/3dPrinter/5_2-DISN-new-master
  source isosurface/LIB_PATH

  nohup python -u preprocessing/create_point_sdf_grid.py --thread_num {recommend 9} --category {default 'all', but can be single category like 'chair'} &> log/create_sdf.log &

  nohup python -u preprocessing/create_point_sdf_grid.py --thread_num 9 --category 'chair' &> log/create_sdf.log &
  nohup python -u preprocessing/create_point_sdf_grid.py --thread_num 9 --category 'chair' &> log/create_sdf_$(date +%Y%m%d_%H%M%S).log &
  python -u preprocessing/create_point_sdf_grid.py --thread_num 9 --category 'chair'
  
  ## SDF folder takes about 9.0G, marching cube obj folder takes about 245G
  
  ```
用 marching cube 重建出来的 1个 reconstructed ground truth model:  
<img width="1803" alt="Screenshot 2024-07-14 at 2 32 47 AM" src="https://github.com/user-attachments/assets/6c4be764-899a-4e6b-935b-1c1d189a48ae">

### 4. 下载和生成 2d image h5 files:
#### 方法1: 下载 2d image following 3DR2N2[https://github.com/chrischoy/3D-R2N2], please cite their paper if you use this image tar file:
  
  ```
  wget http://cvgl.stanford.edu/data2/ShapeNetRendering.tgz
  untar it to {your rendered_dir}
  ```
#### 方法2: 用代码生成 .h5 file generation (about 26 GB) :
  
  ```
  cd {DISN}
  nohup python -u preprocessing/create_img_h5.py &> log/create_imgh5.log &
  ```

##  Camera parameters estimation network

* ### train the camera parameters estimation network:
  ```
  ### train the camera poses of the original rendered image dataset. 
    nohup python -u cam_est/train_sdf_cam.py --log_dir cam_est/checkpoint/{your training checkpoint dir} --gpu 0 --loss_mode 3D --learning_rate 2e-5 &> log/cam_3D_all.log &
   
  ### train the camera poses of the adding 2 more DoF augmented on the rendered image dataset. 
    nohup python -u cam_est/train_sdf_cam.py --log_dir cam_est/checkpoint/{your training checkpoint dir} --gpu 2 --loss_mode 3D --learning_rate 1e-4 --shift --shift_weight 2 &> log/cam_3D_shift2_all.log &
 
* ### if use new rendered 2d dataset:
  ```
  ### train the camera poses of the new rendered image dataset. 
   nohup python -u cam_est/train_sdf_cam.py --log_dir cam_est/checkpoint/{your checkpoint dir} --gpu 0 --loss_mode 3D --learning_rate 1e-4 --src_h5_dir {your new rendered images' h5 directory} --img_h 224 --img_w 224 &> log/cam_3D_easy.log & 
    
  ```
* ### create h5 file of image and estimated cam parameters:
  ```
  ＃＃＃　Create img_h5 to {renderedh5_dir_est} in your info.json, the default is only generate h5 of test images and cam parameters(about 5.3GB) 
  nohup python -u train_sdf_cam.py --img_h5_dir {renderedh5_dir_est} --create --restore_model checkpoint/cam_3D_all --log_dir checkpoint/{your training checkpoint dir} --gpu 0--loss_mode 3D --batch_size 24 &> log/create_cam_mixloss_all.log &
  ```
  
## SDF generation network:

* ### train the sdf generation with provided camera parameters:

  if train from scratch, you can load official pretrained vgg_16 by setting --restore_modelcnn; or you can  --restore_model to your checkpoint to continue the training):

  * support flip the background color from black to white since most online images have white background(by using --backcolorwhite)
  * if use flag --cam_est, the img_h5 is loaded from {renderedh5_dir_est} instead of {renderedh5_dir}, so that we can train the generation on the estimated camera parameters
  ```
  nohup python -u train/train_sdf.py --gpu 0 --img_feat_twostream --restore_modelcnn ./models/CNN/pretrained_model/vgg_16.ckpt --log_dir checkpoint/{your training checkpoint dir} --category all --num_sample_points 2048 --batch_size 20 --learning_rate 0.0001 --cat_limit 36000 &> log/DISN_train_all.log &
  ```

* ### inference sdf and create mesh objects:

  * will save objs in {your training checkpoint dir}/test_objs/{sdf_res+1}_{iso}
  * will save objs in {your training checkpoint dir}/test_objs/{sdf_res+1}_{iso}
  * if use estimated camera post, --cam_est, will save objs in {your training checkpoint dir}/test_objs/camest_{sdf_res+1}_{iso}
  * if only create chair or a single category, --category {chair or a single category}
  * --sdf_res control the resolution of the sampled sdf, default is 64, the larger, the more fine-grained, but slower.
  ```
  source isosurface/LIB_PATH

  #### use ground truth camera pose
  nohup python -u test/create_sdf.py --img_feat_twostream --view_num 24 --sdf_res 64 --batch_size 1  --gpu 0 --sdf_res 64 --log_dir checkpoint/{your training checkpoint dir} --iso 0.00 --category all  &> log/DISN_create_all.log &
  
  #### use estimated camera pose
  nohup python -u test/create_sdf.py --img_feat_twostream --view_num 24 --sdf_res 64 --batch_size 1  --gpu 3 --sdf_res 64 --log_dir checkpoint/{your training checkpoint dir} --iso 0.00 --category all --cam_est &> log/DISN_create_all_cam.log &
  ```
* ### clean small objects:
  * #### if the model doens't converge well, you can clean flying parts that generated by mistakes
  ```
  nohup python -u clean_smallparts.py --src_dir checkpoint/{your training checkpoint dir}/test_objs/65_0.0 --tar_dir checkpoint/{your training checkpoint dir}/test_objs/65_0.0 --thread_n 10 &> log/DISN_clean.log &
  ```

## Evaluation:
### please compile models/tf_ops/ approxmatch and nn_distance and cites "A Point Set Generation Network for 3D Object Reconstruction from a Single Image"
* ### Chamfer Distance and Earth Mover Distance:
  * #### cal_dir specify which obj folder to be tested, e.g. if only test watercraft, --category watercraft
  ```
   nohup python -u test/test_cd_emd.py --img_feat_twostream --view_num 24 --num_sample_points 2048 --gpu 0 --batch_size 24 --log_dir checkpoint/{your training checkpoint dir} --cal_dir checkpoint/{your training checkpoint dir}/test_objs/65_0.0 --category all &> log/DISN_cd_emd_all.log & 
  ```
* ### F-Score caluculation:
  * cal_dir specify which obj folder to be tested, e.g. if only test watercraft, --category watercraft
  also the threshold of true can be set, here we use 2.5 for default:
  ```
   nohup python -u test/test_f_score.py --img_feat_twostream --view_num 24 --num_sample_points 2048 --gpu 0 --batch_size 24 --log_dir checkpoint/{your training checkpoint dir} --cal_dir checkpoint/{your training checkpoint dir}/test_objs/65_0.0 --category all --truethreshold 2.5 &> log/DISN_fscore_2.5.log & 
  ```
 * ### IOU caluculation:
    * cal_dir specify which obj folder to be tested, e.g. if only test watercraft, --category watercraft
    * --dim specify the number of voxels along each 3D dimension.

    ```
      nohup python -u test/test_iou.py --img_feat_twostream --view_num 24 --log_dir checkpoint/{your training checkpoint dir} --cal_dir checkpoint/{your training checkpoint dir}/test_objs/65_0.0 --category all --dim 110 &> DISN_iou_all.log &
    ```
