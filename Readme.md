# Malaria Detection Using Deep Learning

## Overview

This project implements a sophisticated deep learning model to detect malaria parasites in microscopic blood smear images. The system employs advanced neural network architectures to classify cell images into two distinct categories:

- **Parasitized**: Cells infected with malaria parasites
- **Uninfected**: Healthy, uninfected cells

The model leverages **Transfer Learning with MobileNetV2** architecture to achieve high accuracy while maintaining computational efficiency and reduced training time.

## Dataset

The dataset comprises high-resolution microscopic images of red blood cells sourced from clinical samples.

### Classes

- **Parasitized**: Malaria-infected red blood cells
- **Uninfected**: Healthy red blood cells

### Dataset Distribution

| Split | Percentage |
|-------|------------|
| Training | 70% |
| Validation | 15% |
| Testing | 15% |

## Data Preprocessing

The following preprocessing techniques are systematically applied to ensure model robustness:

- **Image Resizing**: All images are resized to 224x224 pixels for consistency with MobileNetV2 input specifications
- **Pixel Normalization**: Values normalized to 0-1 range for optimal neural network convergence
- **Data Augmentation**: Applied to enhance model generalization and reduce overfitting

### Data Augmentation Techniques

To improve model generalization and robustness, the following augmentation strategies are employed:

- **Rotation**: Random rotation transformations
- **Horizontal Flip**: Spatial augmentation
- **Zoom**: Scale variations
- **Shear Transformations**: Perspective transformations

## Model Architecture

The project employs **MobileNetV2** as the base architecture with transfer learning techniques. This approach provides:

- Lightweight and efficient architecture
- Pre-trained weights from ImageNet dataset
- Rapid convergence and training efficiency
- Minimal computational resource requirements

## Technical Stack

- **Deep Learning Framework**: TensorFlow/Keras
- **Model Architecture**: MobileNetV2
- **Image Processing**: OpenCV, PIL
- **Data Analysis**: NumPy, Pandas
- **Development Environment**: Python 3.x

## Installation and Usage

Detailed instructions for installation, training, and evaluation will be provided in the project documentation.

## License and Attribution

Copyright © 2026

**Project Contributors:**
- Vansh Singh
- Soumya Ray
- Aditya Raj
- Aarzoo Yadav

This project is developed and maintained by the above contributors. All rights reserved.
