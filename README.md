# [A Feature Representation Technique for Angular Margin Loss]([https://github.com/peteryuX/arcface-tf2](https://github.com/asfakali/FRAML.git))


[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]([https://colab.research.google.com/github/peteryuX/arcface-tf2/blob/master/notebooks/colab-github-demo.ipynb](https://colab.research.google.com/drive/1f_rDNAHdsDz-0M7EXBn1fUb7WBgqJB8K))

:fire: A Feature Representation Technique for Angular Margin Loss :fire:

>  This work proposes a straightforward yet effective feature representation approach for the angular margin loss (Arc) feature. The representation approach makes it possible to detect similarities between distinct characteristics, which is particularly beneficial for many computer vision problems such as face recognition and object tracking. We offer what is perhaps the most comprehensive experimental evaluation of the most recent state-of-the-art face recognition algorithms, ArcFace, using three face recognition benchmarks.

Original Paper: [ICICV2022](https://link.springer.com/chapter/10.1007/978-981-99-2602-2_7)

Offical Implementation: &nbsp; [ArcFace]([https://github.com/deepinsight/insightface](https://github.com/peteryuX/arcface-tf2))

****

## Contents
:bookmark_tabs:

* [Installation](#Installation)
* [Data Preparing](#Data-Preparing)
* [Training and Testing](#Training-and-Testing)
* [Benchmark and Models](#Benchmark-and-Models)
* [References](#References)

## Installation
:pizza:

Create a new python virtual environment by [Anaconda](https://www.anaconda.com/) or just use pip in your python environment and then clone this repository as following.

### Clone this repo
```bash
git clone https://github.com/asfakali/FRAML.git
cd FRAML
```

### Conda
```bash
conda env create -f environment.yml
conda activate arcface-tf2
```

### Pip

```bash
pip install -r requirements.txt
```

****

## Data Preparing
:beer:

All datasets used in this repository can be found from [face.evoLVe.PyTorch's Data-Zoo](https://github.com/ZhaoJ9014/face.evoLVe.PyTorch#Data-Zoo).

Note:

- Both training and testing dataset are "Align_112x112" version.

### Training Dataset

Download [MS-Celeb-1M](https://drive.google.com/file/d/1X202mvYe5tiXFhOx82z4rPiPogXD435i/view?usp=sharing) datasets, then extract and convert them to tfrecord as traning data as following.

```bash
# Binary Image: convert really slow, but loading faster when traning.
python data/convert_train_binary_tfrecord.py --dataset_path="/path/to/ms1m_align_112/imgs" --output_path="./data/ms1m_bin.tfrecord"

# Online Image Loading: convert really fast, but loading slower when training.
python data/convert_train_tfrecord.py --dataset_path="/path/to/ms1m_align_112/imgs" --output_path="./data/ms1m.tfrecord"
```

Note:
- You can run `python ./dataset_checker.py` to check if the dataloader work.

### Testing Dataset

Download [LFW](https://drive.google.com/file/d/1WO5Meh_yAau00Gm2Rz2Pc0SRldLQYigT/view?usp=sharing), [Aged30](https://drive.google.com/file/d/1AoZrZfym5ZhdTyKSxD0qxa7Xrp2Q1ftp/view?usp=sharing) and [CFP-FP](https://drive.google.com/file/d/1-sDn79lTegXRNhFuRnIRsgdU88cBfW6V/view?usp=sharing) datasets, then extract them to `/your/path/to/test_dataset`. These testing data are already binary files, so it's not necessary to do any preprocessing. The directory structure should be like bellow.
```
/your/path/to/test_dataset/
    -> lfw_align_112/lfw
        -> data/
        -> meta/
        -> ...
    -> agedb_align_112/agedb_30
        -> ...
    -> cfp_align_112/cfp_fp
        -> ...
```

****

## Training and Testing
:lollipop:

You can modify your own dataset path or other settings of model in [./configs/*.yaml](https://github.com/peteryuX/arcface-tf2/tree/master/configs) for training and testing, which like below.

```python
# general (shared both in training and testing)
batch_size: 128
input_size: 112
embd_shape: 512
sub_name: 'arc_res50'
backbone_type: 'ResNet50' # or 'MobileNetV2'
head_type: ArcHead # or 'NormHead': FC to targets.
is_ccrop: False # central-cropping or not

# train
train_dataset: './data/ms1m_bin.tfrecord' # or './data/ms1m.tfrecord'
binary_img: True # False if dataset is online decoding
num_classes: 85742
num_samples: 5822653
epochs: 5
base_lr: 0.01
w_decay: !!float 5e-4
save_steps: 1000

# test
test_dataset: '/your/path/to/test_dataset'
```

Note:
- The `sub_name` is the name of the outputs directory used in the checkpoints and logs folder. (make sure to set it unique to other models)
- The `head_type` is used to choose [ArcFace](https://arxiv.org/abs/1801.07698) head or normal fully connected layer head for classification in training. (see more detail in [./modules/models.py](https://github.com/peteryuX/arcface-tf2/blob/master/modules/models.py#L90-L94))
- The `is_ccrop` means doing central-cropping on both trainging and testing data or not.
- The `binary_img` is used to choose the type of training data, which should be according to the data type you created in the [Data-Preparing](#Data-Preparing).


### Training

Here have two modes for training your model, which should be perform the same results at the end.
```bash
# traning with tf.GradientTape(), great for debugging.
python train.py --mode="eager_tf" --cfg_path="./configs/arc_res50.yaml"

# training with model.fit().
python train.py --mode="fit" --cfg_path="./configs/arc_res50.yaml"
```

### Testing

You can download my trained models for testing from [Benchmark and Models](#Benchmark-and-Models) without training it yourself. And, evaluate the models you got with the corresponding cfg file on the testing dataset. The testing code in [./modules/evaluations.py](https://github.com/peteryuX/arcface-tf2/blob/master/modules/evaluations.py) were modified from [face.evoLVe](https://github.com/ZhaoJ9014/face.evoLVe.PyTorch).

```bash
!python test.py --cfg_path="./configs/arc_mbv2_ccrop.yaml" --sim "ssim"
```
Note:
 - ```--sim``` is used for similarity measure. "ssim" for Structural Similarity Index Measure(SSIM), "diff" for normal difference, "corr" for cross-correlation. "fft_corr" for Fourier domain cross-correlation, "cos" cosine similarity. 
### Encode Input Image

You can also encode image into latent vector by model. For example, encode the image from [./data/BruceLee.jpg](https://github.com/peteryuX/arcface-tf2/blob/master/data/BruceLee.jpg) and save to `./output_embeds.npy` as following.

```bash
python test.py --cfg_path="./configs/arc_res50.yaml" --img_path="./data/BruceLee.jpg" --sim "ssim"
```


## References
:hamburger:

Thanks for these source codes porviding me with knowledges to complete this repository.

- Face Analysis Project on tf2 https://github.com/peteryuX/arcface-tf2
- https://github.com/deepinsight/insightface/tree/master/recognition (Official)
    - Face Analysis Project on MXNet http://insightface.ai
- https://github.com/zzh8829/yolov3-tf2
    - YoloV3 Implemented in TensorFlow 2.0
- https://github.com/ZhaoJ9014/face.evoLVe.PyTorch
    - face.evoLVe: High-Performance Face Recognition Library based on PyTorch
- https://github.com/luckycallor/InsightFace-tensorflow
    - Tensoflow implementation of InsightFace (ArcFace: Additive Angular Margin Loss for Deep Face Recognition).
- https://github.com/dmonterom/face_recognition_TF2
    - Training a face Recognizer using ResNet50 + ArcFace in TensorFlow 2.0
