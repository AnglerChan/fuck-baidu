# 试题说明

## 任务描述

基于MMC数据集的图像分类，MMC包含三种模态（RGB、Depth、Infrared）对齐的13种类别的物体，每种类别大约130到170个图像，需要根据图片三类不同模态的特征，用算法从中识别该图像属于哪一个类别。

## 数据说明

任务所使用图像数据集，训练集每个模态包含2000张图片，被分为13类，每个类别图片超过100张；测试集每个模态包含1000张图片，同样包含13个类。

其中13个类别分别为： 0: person、 1: boat、 2: traffic sign、 3: animal、 4: seat、 5: bird、 6: sign、 7: cyclist、 8: bicycle、 9: car、 10: ball、 11: light、 12: garbage can。

已将训练集按照“图片名 标签”的格式放到了train_2k文件夹下的train_labels.txt文件中。 （训练集 “图片名 标签”， 测试集 “图片名”。）

数据集下载链接： https://pan.baidu.com/s/14eQOlRzbVzocZeuraXIIag?pwd=mm01 （提取码: mm01）

*** 狗日的百度网盘 ***

数据集详细介绍以及baseline代码请参考：https://github.com/Sandwich-Lee/Multi-Modal_Object_Classification/tree/master

注意：需要自行把训练数据划分训练集和验证集。

## 提交结果

训练完成后，利用提供的“Inference.py” 文件生成在测试集上的预测结果, 结果文件为csv文件格式，命名为 "submission.csv" ，文件内的字段需要按照以下指定格式写入。

## 结果文件要求：

1.每个类别的行数和测试集原始数据行数应一 一对应，不可乱序。

2.输出结果应检查是否为1000行数据，否则成绩无效。

3.输出结果文件命名为submission.csv，一行一个类别，样例如下：

filename,label_pred

000003_004_00000084_0.png,0

000003_004_00000084_1.png,1

000003_004_00000167_1.png,2

000003_004_00000250_0.png,3

000003_005_00000149_0.png,4

000003_016_00000001_1.png,5

000003_016_00000001_2.png,6

000003_016_00000001_3.png,7

......

In [ ]



# Multi-Modal Object Classification
This is the official repository for the Multi-Modal Classification.

This challenge focuses on Object Classification utilizing multi-modal data source including RGB, depth, and infrared images. You can visit the official website for more details.

# Dataset
In this track, we provide a dataset named MMC (Multi-Modal Object Classification), which comprises 3,000 multi-modal image pairs (2000 for training and 1000 for testing) across 13 classes.

## Example
| Depth | Thermal-IR | RGB |
|:-----------:|:------------:|:---------:|
| ![Depth Output](pics/Depth/000003_002_00000060_0_8.png) | ![IR Output](pics/IR/000003_002_00000060_0_8.png) | ![RGB Output](pics/RGB/000003_002_00000060_0_8.png) |
| ![Depth Output](pics/Depth/000006_008_00000103_0_5.png) | ![IR Output](pics/IR/000006_008_00000103_0_5.png) | ![RGB Output](pics/RGB/000006_008_00000103_0_5.png) |
## Structure
```
MMC
├── train_2k
│ ├──color
│ │ ├── train_0001.png
│ │ ├── train_0002.png
│ │ ├── ... ...
│ │ ├── train_4000.png
│ ├──depth
│ │ ├── train_0001.png
│ │ ├── train_0002.png
│ │ ├── ... ...
│ │ ├── train_4000.png
│ ├──infrared
│ │ ├── train_0001.png
│ │ ├── train_0002.png
│ │ ├── ... ...
│ │ ├── train_4000.png
│ │ ├── ... ...
| |——train_labels.txt
├── test
│ ├──color
│ │ ├── test_0001.png
│ │ ├── test_0002.png
│ │ ├── ... ...
│ │ ├── test_1000.png
│ ├──depth
│ │ ├── test_0001.png
│ │ ├── test_0002.png
│ │ ├── ... ...
│ │ ├── test_1000.png
│ ├──infrared
│ │ ├── test_0001.png
│ │ ├── test_0002.png
│ │ ├── ... ...
│ │ ├── test_1000.png
```
# Baseline
This code is based on Resnet18.
- **❗Note!!!** The validation set is not provided, you should divide the train set appropriately by yourself to validate during training.
- We have modified the model to accommodate this multimodal task, while you can also build your own model to accomplish this task
# Training
- Change the root path of the dataset (your path to train_2k)
- run `tain.py`
Train your own model:
```
  python train.py --root path_to_train_2k \
    --train_labels train_labels.txt \
    --val_labels val_labels.txt \
    --epochs 80 \
    --eval_period 1 \
    --batch 64 \
    --num_classes 13 \
    --output_file path_to_save_the_log_file_and_the_model \

```
# Testing
Generate the predictions `submission.csv` for test set:
run `Inference.py`
 ```
  python Inference.py --root path_to_test_2k \
    --checkpoint path_to_the_best_model \
    --save path_to_save_the_submissionfile \
    --num_classes 13
 ```
- **❗Note** Results `submission.csv` will be generated automatically and it's the only file you need to submit to the platform for evaluation. 
