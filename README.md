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
- run `tain.sh`
Train your own model:
```
  sh train.sh
```
# Testing
Generate the predictions `submission.csv` for test set:
run `Inference.sh`
 ```
  sh Inference.sh
 ```
- **❗Note** Results `submission.csv` will be generated automatically and it's the only file you need to submit to the platform for evaluation. 
