# PyTorch Projects

This repo contains a lot of assignments and personal projects I did during the Packt Pytorch Specialization in coursera. For the sake of simplicity we have broken them into two parts.

Collectively, both notebook collectively cover following projects:

1. Car Dataset Regression - Linear Layer based DNN
2. Iris Classification - Linear Layer based DNN
3. Concrete Crack Detector - ResNet based CNN Classifier
4. Dog Breed Classifier - ResNet based CNN Classifier and ViT FineTuning
5. Heart Beat Sounds Anomaly Detection - Audio Classification using Spectrograms and CNN: 
6. Fruit Detection - Object Detection  
7. Face Mask Detection - YoloV8 Object Detection
8. Pet Classification - DenseNet Classifier
9. Style Transfer - CNN Style Transfer
10. Flight Passenger Modeling - LSTM based Time Series Modeling
11. Variational AutoEncdoer (VAE) for image compression
12. Fake Celebrity Face Generator using VAE-GAN

Among above following are personal projects:
1. Fake Celebrity Face Generator: https://www.kaggle.com/datasets/jessicali9530/celeba-dataset
2. Concrete Crack Classifier: https://www.kaggle.com/datasets/arnavr10880/concrete-crack-images-for-classification
3. Heart Beat Sounds Anomaly Detection: https://www.kaggle.com/datasets/kinguistics/heartbeat-sounds/discussion?sort=undefined
4. Face Mask Detection: https://www.kaggle.com/datasets/andrewmvd/face-mask-detection
5. Dog Breed Classifier: images sourced from web

## Fake Celebrity Face Generator

In this project we used over 200k+ celebrity face dataset to train a custom VAE-GAN model. We finetuned the model in favor of generator to create credible fake images, all trained on local hardware.

<img width="581" height="523" alt="image" src="https://github.com/user-attachments/assets/b42a288f-a00e-4859-bc17-91c6bbde913e" />

Since we trained a 64,64 shaped model it may appear blurry without upscaling but this fake image is very similar to training dataset. We can compare them below better:  

<img width="1058" height="134" alt="recon" src="https://github.com/user-attachments/assets/723e5292-bf9c-4df8-800e-0ce2a8c4aa46" />

In above image we took batch celebrity images and obtained their latent distribution in form of mu and log variance and use that data to create the image instead of using random noise and we can see that generator was able to learn the latent distributions properly as it was able to add proper variations to the images, like feminizing some source images and even changing race for some.

## Concrete Crack Classifier

In this project we created ResNet style based CNN solution for detection Concrete Crack and obtained 97% test accuracy.

<img width="1007" height="375" alt="image" src="https://github.com/user-attachments/assets/8349a6e2-0a77-4fdd-ae4f-a47baa84174b" />

And on basis of GradCAM visuals we can observe that model was able to properly focus on the cracks.

## Heart Beat Sounds Anomaly Detection

In this project we used kaggle dataset to solve two problems, first create a model for heart beat sound segmentation which can identify which device is generating some specific sound, and other was to classify heart beat sounds across four categories such as Normal, ExtraHLS, Noise, and murmurs.

We first converted sound files into spectrogram using torchaudio and then created a CNN based solution with two heads, and obtained 100% test accuracy in device segmentation task and 80% accuracy in classification. We could significantly improve the performance of the model by finetuning a pretrained model.

<img width="646" height="500" alt="image" src="https://github.com/user-attachments/assets/12c05c09-f47a-454b-9a44-18f6d5fcc297" />

We trained model on mel spectrograms like above.

## Face Mask Detection

In this project we created a YoloV8 Face Mask detector using the Ultralytics library and PyTorch. We obtained a mAP50@0.5 of 0.812 over three classes, and 90% recall in wearing mask class.

<img width="702" height="457" alt="image" src="https://github.com/user-attachments/assets/ce40f52f-002e-4f5c-8395-19a29a2175bd" />

## Dog Breed Classifier

In this project we created a custom ResNet-18 like model for dog breed classification over 3 dog breeds. We scraped data from net and obtained 91% test accuracy with custom model. Then we used timm libary to finetune a vit_base_patch16_224 Vision Transformer (ViT) and obtained 100% test accuracy. Following image reflects gradcam focus on the image features, ViT model was able to focus on many disntinguishing features acrossd different dog breeds compared to ResNet based model which was focusing only on earts and mouth.

<img width="1255" height="617" alt="image" src="https://github.com/user-attachments/assets/c4811788-0bbd-4e49-a8f2-9f0c184204ee" />
