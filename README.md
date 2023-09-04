# NN-CA3

### 1.Introduction-to-Transfer-Learning [Link](#part-1-introduction-to-transfer-learning)

### 2.Faster-RCNN-Network-Architecture [Link](#part-2-faster-rcnn-network-architecture)

# Part 1: Introduction to Transfer Learning

## Dataset

For this part, we implemented the project using the EuroSAT dataset, which can be found here: [EuroSAT Dataset](https://github.com/phelber/EuroSAT).

## Results Report

Using transfer learning, we achieved an accuracy of approximately 88%, which indicates that with transfer learning, although the accuracy is slightly lower compared to training a new model from scratch, we reached the solution much faster. The F1-score was approximately 88%, and the precision reached around 89%. All these values remained fairly consistent for training and evaluation data after 40 epochs.

## Training Parameters

For the training section, the parameters were set as follows, following the recommendations from the research paper:

- Batch size = 32
- Momentum = 0.9
- Learning rate = 0.001
- Epochs = 40
- Dropout rate = 0.5

## Network Architecture

The recommended model architecture based on the research paper is as follows:

- The CNN layer starts with a 32x32 input.
- Layers of 64*, 64 128*, 128 256*, 256, and 512*512 are added successively.
- After each layer, a max-pooling layer of size 2x2 is added.
- After each convolutional layer, a non-linear layer is added to capture features as the features are not well-defined, and pixels are very small.
- Before each convolutional layer, a zero-padding layer is added, which uses the ReLU activation function to increase non-linearity.
- In this model, three layers of 16VGG are transformed into two layers to reduce the number of parameters and increase speed. The first fully connected (fc) layer has 4096 channels, the second has 1000 channels, and the final layer is a softmax layer.

## Model Features

This network uses fewer parameters and can fit faster compared to the larger 16VGG network. It has achieved an accuracy of 95 for pixel low-ultra data. This network consumes less memory and can effectively detect low-level features.

## Model Limitations

One limitation of this model is its specificity to the particular dataset used, which is for images with pixel low-ultra. Other models may perform better on different datasets. Additionally, it took considerably less time to reach 88% accuracy using transfer learning compared to the 25 hours mentioned in the paper.

## Data Preprocessing

The preprocessing steps recommended by the research paper are as follows:

- Resize each image to 64x64.
- Flatten the image after passing through convolutional layers.
- Convert the flattened vector into an array, and display its apparent features using string operations.
- Finally, scale each pixel in the image by dividing by 225.

## Data Split

The data is split into 80% training and 20% testing sets for evaluation.

## Recognizable Image Types

This project focuses on images with very small pixels (pixel low-ultra) which required expanding the dataset; otherwise, we would have obtained lower accuracy. These data have limited feature extraction because they are pixel low-ultra images. Additionally, the low-level features are extracted using zero-padding to increase non-linearity.

## Handling Missing Data

If there is no data available in a category, it poses a problem for the CNN model as it has no data to learn from that category. In such cases, you can collect some data for that category and then fit the model. If there are only a few data samples available for a category, data augmentation techniques like vertically or horizontally flipping, cropping, transposing, rotating, shifting, etc., can be used to compensate for the limited data. If none of these methods work, you may consider removing that category.

## Data Inspection

The dataset provided contains 10 classes, with each class containing 3000-2000 images, resulting in a total of 27,000 images. All images are of size 64x64.

## Classification Report

Here is the classification report for our model:
precision recall f1-score support

       0       0.91      0.89      0.90       600
       1       0.92      0.89      0.91       600
       2       0.90      0.87      0.88       600
       3       0.75      0.86      0.80       500
       4       0.93      0.95      0.94       500
       5       0.71      0.90      0.80       400
       6       0.90      0.75      0.82       500
       7       0.94      0.95      0.94       600
       8       0.83      0.76      0.79       500
       9       0.96      0.93      0.94       600

# Part 2: Faster RCNN Network Architecture

### Feature Extractor

The first component of the CNN-R Faster model is a deep convolutional neural network (CNN) used as a shared feature extractor. The authors of this architecture use the -16VGG architecture as the base CNN, and they replace the first fully connected layer with a convolutional layer to accommodate images of various sizes. Other base networks like ResNet can also be used. The CNN is pretrained on the ImageNet dataset, allowing it to learn general visual features that can be useful for object detection.

### Region Proposal Network (RPN)

Another component is the Region Proposal Network (RPN), which is a fully convolutional network that takes the feature maps (output of the CNN) as input and generates region proposals for potential objects of interest. The RPN consists of three convolutional layers along with two parallel fully connected layers. One is responsible for calculating bounding box coordinates, and the other is for classifying the importance of objects within those regions. The convolutional layers use 3x3 kernels with stride 1 and padding 1 to maintain the geometric resolution of the input feature maps. The RPN generates a set of bounding boxes with fixed sizes for each region and predicts objectness scores and offset ranges for each region. Objectness scores are calculated as binary classification scores using the softmax function, and object boundaries are predicted through regression using the 1L loss.

### Object Detector

The final component, the object detector, takes the regions of interest (ROIs) generated by the RPN as input and performs object classification and regression to refine the ROIs. This detector is based on the -16VGG architecture, replacing the fully connected layers with two parallel layers: one for classification and the other for regression. ROIs are first transformed into fixed-size feature maps using ROI pooling layers. Then, these pooled features are passed through fully connected layers to predict improved regions and class scores.

In summary, the CNN-R Faster model is a complex neural network structure with intensive computations, requiring precise parameter tuning. Nevertheless, it has shown good performance on various benchmark datasets.

## Training the Network

To train the CNN-R Faster model, you can follow these steps:

1. Download the relevant dataset from the provided link and upload it to Google Drive for later use in Google Colab.
2. Prepare the dataset and create data loaders for the training, evaluation, and test sections. You can use the `PASCALDataset` function for this purpose.
3. Set the batch size parameter to 2 for training data, as larger values may fill the GPU memory and smaller values have been found to yield better results in our experiments.
4. Load the `fpn_50resnet_fasterrcnn` model and use `FastRCNNPredictor` as the box predictor.
5. Move the model to the GPU for faster training and testing.
6. Use the SGD optimizer with a learning rate of 0.005 and momentum of 0.9.
7. Apply a learning rate scheduler that reduces the learning rate by half after each epoch, as this configuration yielded the best results in our experiments.
8. Train the model for 5 epochs using the `epoch_one_train` function.

After training, you can save the model for later use.

## Object Detection and Counting

You can visualize the model's output on three test dataset images and additional custom images for object detection and counting.

Please note that specific code implementations and visualizations for the outputs are typically done in code environments like Jupyter notebooks. You can refer to the notebook associated with this README for code details and visualizations.
