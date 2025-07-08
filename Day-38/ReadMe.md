# TorchVision : Image Classification Using Pre-Trained Models - DAY 38

In this notebook, we will learn how to use pre-trained models to perform image classification.Thanks to the ImageNet project, pre-trained models are available in torchvision that have been trained to detect objects from 1,000 different classes. With just a few lines of code, we will learn how to use these pre-trained models out-of-the-box to perform image classification with no training required.

## 1. ImageNet and ILSVRC

The ImageNet project is a large visual database designed for visual object recognition software research. The idea for this project was conceived over 15 years ago by AI researcher Fei-Fei Li. The ImageNet team presented their dataset for the first time in 2009.

Since 2010, the ImageNet project runs an annual software competition where research teams evaluate their computer vision algorithms for various visual recognition tasks such as Object Classification and Object Localization. The training data is a subset of ImageNet with 1.2 million images belonging to 1,000 classes. Deep Learning came to the limelight in 2012 when Alex Krizhevsky and his team won the competition by a margin of a whopping 11%. ILSVRC and Imagenet are sometimes used interchangeably.

There are various subsets of the ImageNet dataset used in various contexts. One of the most highly used subsets of ImageNet is the "ImageNet Large Scale Visual Recognition Challenge (ILSVRC) 2012-2017 image classification and localization dataset". This is also referred to in the research literature as ImageNet-1K or ILSVRC2017, reflecting the original ILSVRC challenge that involved 1,000 classes.

## 2. Pre-Trained Classification Models in Torchvision

The winners of ILSVRC have been very generous in releasing their models to the open-source community. Many models are available in Torchvision, such as AlexNet, VGGNet, Inception, ResNet, Xception, and many more. Apart from the ILSVRC winners, many research groups also share their models, which they have trained for similar tasks, e.g., MobileNet, SqueezeNet, etc. All the models trained on ImageNet are for classifying images into one of 1,000 classes.

Torchvision comes bundled with many pre-trained classification models. As of torchvision version 0.19, there are 20 different pre-trained models available, where some versions contain many variants as well. The list of models can be found here. Here we will use the following pre-trained models to make predictions on several sample test images.

AlexNet
VGG16
ResNet18
To use any of the pre-trained models in Torchvision, there are four basic steps required:

**Load a pre-trained model**
**Preprocess the input image(s) using transforms.**
**Forward pass the image to model to generate predictions.**
**De-code the predictions and map the classnames and class_ids on post-processing.**

## 3. Pre-trained Model Setup

###  Preprocess the Inputs
When these models were trained on the ImageNet dataset, the input images were preprocessed in a specific way. Besides resizing images to conform to the expected size of the network, the images are typically zero-centered and normalized. When using these models, it's important that your input images are pre-processed in the same way the training images were processed. For convenience, each model in torchvision includes a transforms.

Transforms are common image transformations. They can be chained together using Compose.
```python
# Specify image transformations.
transform = transforms.Compose([
                  transforms.Resize(256),     #Resize the image to 256×256 pixels.
                  transforms.CenterCrop(224), #Crop the image to 224×224 pixels about the center.
                  transforms.ToTensor(),      #Convert the image to PyTorch Tensor data type.
                  transforms.Normalize(
                  mean=[0.485, 0.456, 0.406], #Normalize the image with imagenet mean and std.
                  std=[0.229, 0.224, 0.225]
                  )])
```

### Instantiate the Model
Here we will use the ResNet18 model to describe the approach. Here we call the built-in model resnet18() to instantiate the ResNet18 pre-trained model. Notice that the function has several optional arguments, which provide a lot of flexibility for using the model. However, the default settings allow you to use the model right out of the box to perform image classification from 1,000 classes in the ImageNet dataset.

```python
# Load resnet18 model
model = models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1) #models.resnet18(weights = "DEFAULT")
```

###  Forward Pass
After pre-processing the input images, we can then forward pass them to the model as shown below. Because PyTorch process image data in batches, we will need to add a batch dimension to the images, even if we process one image at a time. As an example, ResNet18 expects color images with a shape of [3,224,224], but we must add a batch dimension so that the image batch has a shape: [B, C, H, W], even if we intend to process a single image at a time. We'll see how this is done further below.

```python
img_t = transform(img)
batch_t = torch.unsqueeze(img_t, 0) #Add batch dimension [C,H,W] --> [B,C,H,W]
```

### Prediction
In PyTorch, we can decode the predictions returned by the model to get the top k predictions. The process involves sorting the output probabilities in descending order and then applying a softmax function to get the prediction percentages.

- Use torch.sort to sort the output probabilities in descending order.
- Apply torch.nn.functional.softmax to convert the output logits to probabilities.
- Extract the top k predictions along with their class IDs and class descriptions.

There are 1,000 classes in the ImageNet dataset, and even though these are all very clear and distinct images, many share characteristics with other classes. Consequently, the model's top k predictions might still be relevant to the inference image, even if they are not the exact class.

For eg. the second highest probability prediction of magnetic compass image was stopwatch, which is relevant due to their shared visual features.