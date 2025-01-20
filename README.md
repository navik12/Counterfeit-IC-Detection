# Counterfeit IC Detection System

## Overview
This project develops a Convolutional Neural Network (CNN) to detect counterfeit integrated circuits (ICs). The proposed model combines techniques like data augmentation and early stopping, achieving a validation accuracy of **94%**. It outperforms AlexNet in accuracy and VGG16 in speed and complexity.

## Features
- **Custom CNN Model**:
  - 8 convolutional layers with batch normalization and ReLU activation.
  - Optimized for a balance between complexity and performance.
- **Data Augmentation**:
  - Random flips, rotations, and image resizing for better generalization.
- **Evaluation Metrics**:
  - Accuracy, validation loss, and confusion matrix for performance evaluation.

## How It Works
1. **Dataset**:
   - Images labeled as "Approved" or "Counterfeit."
   - Original dataset augmented to increase training data size.
2. **Model Training**:
   - Implemented using PyTorch.
   - Utilizes Cross-Entropy loss and SGD optimizer.
   - Early stopping prevents overfitting.
3. **Model Evaluation**:
   - Validation accuracy: **94%**.
   - Successfully detects 6 out of 7 counterfeit ICs in test data.

## How to Run
1. Install dependencies:
   - Python 3.x
   - PyTorch, NumPy, Matplotlib
2. Prepare data:
   - Place images in respective "Approved" and "Counterfeit" folders.
   - Use the provided code to preprocess and augment the dataset.
3. Train the model:
   - Run the training script (`execution.ipynb`) to train and evaluate the model.
   - View accuracy and loss plots for performance insights.

## Results
- **Proposed Model**:
  - Validation Accuracy: **94%**
  - Faster and simpler compared to VGG16.
- **AlexNet**:
  - Validation Accuracy: **58.8%**
- **VGG16**:
  - Validation Accuracy: **70%**

## Future Work
- Enhance the dataset with more images and different augmentations.
- Add dropout layers for improved generalization.
- Experiment with hyperparameter tuning and advanced architectures.

## References
- Key techniques like AlexNet and VGG16.
- Data augmentation and CNN advancements in counterfeit detection.
