# Fine-tuning Pretrained Models - DAY 40
In the previous notebook, we learned how Transfer Learning to train a Convolutional Neural Network (CNN).

## Quick Refresher
A typical image classification architecture consists of 4 parts
- Input image
- Feature Extractor - a bank of convolutional layers that extract useful features for classification.
- Classifier - a bank of fully connected layers that classify the image into its output classes.
- Output vector of class probabilities

In this notebook, we will fine-tune a Mobilenet v3 small model with ImageNet pretrained weights on the same 10 Monkey Species dataset.
We will understand the significance of fine-tuning a pretrained model to adapt to a dataset that has shared features with ImageNet.

## Dataset and Training Configuration Parameters
Before we describe the model implementation and training, we’re going to apply a little more structure to our training process by using the dataclasses module in python to create simple DatasetConfig and TrainingConfig classes to organize several data and training configuration parameters. This allows us to create data structures for configuration parameters. The benefit of doing this is that we have a single place to go to make any desired changes.

### Load Custom Datasets in PyTorch
Till now, we have experimented with datasets like Fashion MNIST available with the PyTorch Torchvision library.

In this notebook, we will see how to load raw images present in a folder.

In the real world, we have need the manage the structure and preprocessing of the dataset on our own.

To illustrate a few preprocessing, we have chosen the 10 Monkey Species dataset from Kaggle. You can download the data from here. You need to extract data. We have already uploaded the extracted data in the lab.

Each folder contains 10 subforders labeled as n0~n9, each corresponding a species form Wikipedia's monkey cladogram. Images are 400x300 px or larger and JPEG format (almost 1400 images). Images were downloaded with help of the googliser open source code.

**Label mapping**:
Label	|Monkey Species
n0	    |alouatta_palliata
n1	    |erythrocebus_patas
n2	    |cacajao_calvus
n3	    |macaca_fuscata
n4	    |cebuella_pygmea
n5	    |cebus_capucinus
n6	    |mico_argentatus
n7	    |saimiri_sciureus
n8	    |aotus_nigriceps
n9	    |trachypithecus_johnii

## Fine-tuning with Pretrained Model
We will use mobilenet_v3_small from torchvision classification models. MobileNetV3 Small known for its lightweight architecture and efficient performance is crucial for mobile and edge applications. It offers significant improvements in speed and accuracy, making it ideal for real-time computer vision tasks on resource-constrained devices.

We conclude that it is often beneficial to start with fine-tuning pretrained models rather than building a model from scratch. Fine-tuning leverages the learned features of large datasets like ImageNet, leading to faster convergence and often better performance, especially when working with smaller datasets. This approach not only saves time but also improves the overall efficiency of the training process.

By adopting a fine-tuning strategy, you can achieve high accuracy with less computational effort and fewer resources, making it a practical choice for your image classification tasks.