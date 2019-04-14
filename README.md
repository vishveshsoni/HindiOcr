# HindiOcr

![](https://img.shields.io/github/tag/pandao/editor.md.svg)

Ocr that extract character from the images.
Currently it can able to recognize language- ENGLISH and HINDI

Application - Flasked based server get images from mobile devices and return extracted text in json formate.

##### WITH TOTAL 58 CLASSES : Vowels, Consonants and Digits


## 1. Image Processing

------------
### -   Character localization on real world image 

![](https://i.ibb.co/1vgQykz/Capture.png)

------------
### -   Character localization on Document image 

![](https://i.ibb.co/xHhZz31/contours.jpg)

------------

![](https://i.ibb.co/6DNNgMw/bin-1.png)

## 2. Deep Learning
### Architecture
##### CONV2D --> MAXPOOL --> CONV2D --> MAXPOOL -->FC -->Softmax--> Classification

### Python Implementation
##### 1. Dataset- Devnagari Character Dataset.
##### 2. Images of size 32 X 32.
##### 3. Convolutional Network (CNN).

------------


### Train Acuracy ~ 97%
### Test Acuracy ~ 92%


------------

[![Output](https://s3.amazonaws.com/sportsseam-public-read/NFL/demo/wKx92kXT4e.gif "Output")](https://s3.amazonaws.com/sportsseam-public-read/NFL/demo/wKx92kXT4e.gif "Output")
