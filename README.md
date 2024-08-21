# Teeth-Disease-Recognition

Creating a README file for your GitHub repository is an excellent way to document your project, making it accessible and understandable to other developers and stakeholders. Here's a template for a README that outlines your project, focusing on key elements and leaving placeholders for detailed results and graphical representations.

### README.md Template for TeethNet Project

---

## TeethNet: Deep Learning for Teeth Classification

### Project Overview
TeethNet is a convolutional neural network designed to classify dental images into various categories of teeth. This model is built using PyTorch and tested using a custom dataset that includes images of different types of teeth prepared for machine learning applications.

### Data Preparation

#### Dataset
The dataset comprises images of teeth, categorized by type. Each category is stored in its own subdirectory within the dataset's main directory, facilitating easy loading and processing.

#### Image Transformations
To ensure robustness and improve generalization, the following transformations are applied:
- **Resize**: Images are resized to 150x150 pixels.
- **Random Horizontal Flip**: Augments the dataset by randomly flipping images horizontally.
- **ToTensor and Normalize**: Converts images to PyTorch tensors and normalizes their pixel values.

### Model Architecture: TeethNet

TeethNet is constructed using PyTorch and features a combination of convolutional layers, pooling, and fully connected layers tailored for effective image classification:

- **Convolutional Layers**: The network starts with three convolutional layers that progressively increase the number of filters from 32 to 128. Each layer uses a 3x3 kernel with padding of 1 to maintain spatial dimensions.
  - `conv1`: 32 filters
  - `conv2`: 64 filters
  - `conv3`: 128 filters

- **Pooling**: Each convolutional layer is followed by a max pooling layer with a 2x2 window to reduce the spatial dimensions by half, thereby condensing the image features.

- **Activation Functions**: ReLU activation functions are applied after each convolutional operation to introduce non-linearities into the model, helping it learn more complex patterns.

- **Fully Connected Layers**: After the convolutional and pooling layers, the network flattens the output and feeds it into two fully connected layers:
  - `fc1`: Transforms the flattened features into a 512-dimensional space.
  - `fc2`: Maps the 512 features to 7 output classes, corresponding to the different types of teeth.

- **Output**: The final output layer uses softmax (not shown directly in the architecture but implied for classification tasks) to derive probabilities for each of the seven classes.

This architecture is designed to efficiently process and classify images based on learned features that capture essential characteristics of different teeth types.



### Training

The model is trained using stochastic gradient descent (SGD) with cross-entropy loss. Key training parameters include:
- **Optimizer**: SGD with configurable learning rate and momentum.
- **Epochs**: Model training is configurable but typically set for 10 epochs.

### Results and Evaluation

Model performance is evaluated on a validation set after training, with accuracy as the primary metric. Model parameters are saved post-training for deployment and further testing.

#### Results Table

| Learning Rate | Momentum | Batch Size | Optimizer Type | Accuracy |
|---------------|----------|------------|----------------|----------|
| 0.01          | 0.9      | 32         | SGD            | 76%      |
| 0.001         | 0.99     | 32         | SGD            | 55%      |
| 0.001         | 0.9      | 64         | Adam           | 86%      |
| 0.001         | 0.9      | 128        | Adam           | 83%      |
| 0.01          | 0.9      | 64         | SGD            | 90%      |


### Graphical Representations

- **B. Accuracy vs. Epoch**: Graph showing how accuracy changes with each epoch during training.
- **A. Loss vs. Epoch**: Graph showing changes in training and validation loss over epochs.
  
  ![Accuracy Tracing](https://github.com/Hassouna9/Teeth-Disease-Recognition/blob/main/graph.png)

### Model Evaluation and Saving

After validation, the model's accuracy is calculated, and its parameters are saved to `TeethModel.pth` for future use. This ensures that the model can be deployed or further developed without the need for retraining.

### Usage

Instructions on how to load the model and make predictions on new data should be included here.

### Conclusions

Summary of the project findings, model's strengths, limitations, and potential applications in dental imaging and diagnosis.
