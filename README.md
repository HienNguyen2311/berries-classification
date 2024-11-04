# Fruit Image Classification using Deep Learning
This project implements and compares different deep learning models for classifying images of fruits (cherry, strawberry, tomato) using convolutional neural networks (CNNs) and transfer learning.

## Features
* Exploratory data analysis of fruit image dataset
* Baseline MLP model implementation
* Comparison of different hyperparameters:
    * Loss functions (Cross Entropy vs Negative Log-Likelihood)
    * Batch sizes (32, 64, 100, 128)
    * Optimizers (Adam, AdaGrad, RMSProp)
    * Regularization strategies (Dropout, L2)
* Transfer learning using pre-trained AlexNet
* Visualization of model performance and comparisons

## Installation
1. Clone this repository
2. Install required dependencies:
```
pip install torch torchvision matplotlib seaborn pandas numpy pillow
```

## Usage
Run the Jupyter notebook `capstone_project.ipynb`

## Data
The dataset contains 4500 images of fruits:
* 1500 cherry images
* 1500 strawberry images
* 1500 tomato images

Images are split 80/20 into training and test sets.

## Technologies Used
PyTorch, torchvision, matplotlib, seaborn, pandas, numpy, Pillow

## Sample Visualizations

Models comparison

![project model comparison](https://github.com/user-attachments/assets/1120ec74-7213-44cc-964a-fa44f34ba5db)

## License
This project is licensed under the MIT License.

## Project Report
The project report provides a detailed analysis of the fruit image classification problem and the deep learning approaches used. Key points from the report include:
* Dataset consists of 4500 images (1500 each of cherry, strawberry, tomato), split 80/20 into training and test sets
* Exploratory data analysis performed on image sizes, color distributions, and variety of images
* Data preprocessing includes resizing, normalization, and data augmentation techniques like rotation and flipping
* Custom CNN model built from scratch is compared with a baseline MLP model and pre-trained AlexNet using transfer learning
* Hyperparameter tuning experiments conducted on loss functions, batch sizes, optimizers, and regularization methods
* Transfer learning with AlexNet achieves the best performance at 84.6% test accuracy
* CNN is concluded to be the best approach for image classification tasks compared to MLP
