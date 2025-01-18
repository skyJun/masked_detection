# Mask Detection with ResNet50 and EfficientNet

## Introduction

This project focuses on developing a mask detection model to classify whether a person is wearing a mask or not. The initiative was driven by the need to ensure compliance with public health guidelines during the COVID-19 pandemic. By utilizing deep learning models such as ResNet50 and EfficientNet, we aim to achieve accurate and efficient mask detection.

---

## Features

- **Binary Classification**: Determines if a person is wearing a mask.
- **Advanced Architectures**: Implements ResNet50 for residual learning and explores EfficientNet for enhanced performance.
- **Data Augmentation**: Utilizes image transformations to increase dataset diversity.
- **Optimization Techniques**: Applies weight decay and pretraining for improved generalization.

---

## Dataset

1. **Data Collection**:
   - Synthetic mask overlay on facial images using MaskTheFace.
   - Generated a dataset of 4,000 images, balanced for both classes.

2. **Data Augmentation**:
   - Techniques: Random rotation, flipping, and cropping.
   - Purpose: Increase diversity and improve learning performance.

---

## Models

### 1. **ResNet50**
   - A deep residual network consisting of 50 layers.
   - Solves the degradation problem using residual learning with identity mappings.
   - Pretrained on ImageNet for faster convergence.

### 2. **EfficientNet**
   - Employs a compound scaling method for depth, width, and resolution.
   - More computationally efficient than ResNet50 while achieving higher accuracy.

---

## Optimization Techniques

- **Weight Decay**: Prevents overfitting by keeping model weights small.
- **AdamW Optimizer**: Decouples weight decay from gradient updates for consistent regularization.
- **Pretraining**: Leverages ImageNet-trained ResNet50 for feature extraction.

---

## Performance Evaluation

- Models were trained with varying hyperparameters (`λ_norm` values).
- ResNet50 showed stable learning and validation accuracy.
- EfficientNet exhibited better accuracy but required more data and epochs for optimal performance.

---

## Results

- **ResNet50**:
  - Validation Accuracy: ~76%.
  - Fast training due to smaller parameters but limited in handling diverse data.

- **EfficientNet**:
  - Superior performance in training metrics with potential for improvement with more data.

---

## Limitations

1. Limited dataset size for real-world evaluation.
2. Challenges in escaping local minima during training.
3. Lower generalization for unseen data.

---

## Usage

### Prerequisites
- Python 3.x
- PyTorch
- Additional libraries: torchvision, numpy, matplotlib

### Training the Model
1. Place the dataset in the `data/` folder.
2. Run the training script:
   ```bash
   python train.py
   ```

### Evaluation
1. Evaluate the model's performance:
   ```bash
   python evaluate.py
   ```

---

## References

1. [MaskTheFace Repository](https://github.com/aqeelanwar/MaskTheFace)
2. He, K., Zhang, X., Ren, S., & Sun, J. (2015). *Deep Residual Learning for Image Recognition*. [arXiv:1512.03385](https://arxiv.org/abs/1512.03385)
3. Tan, Le. (2019). *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks*. [arXiv:1905.11946](https://arxiv.org/abs/1905.11946)
4. Loshchilov, I., & Hutter, F. (2019). *Decoupled Weight Decay Regularization*. [arXiv:1711.05101](https://arxiv.org/abs/1711.05101)

---
