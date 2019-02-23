# HindiOcr

![](https://img.shields.io/github/tag/pandao/editor.md.svg)

Ocr that extract character from the images.
Currently it can able to recognize language- ENGLISH and HINDI

WITH TOTAL 46 CLASSES : vowels, consonants and digits

Application is use python server get image from mobile devices, detect, extract the character and send text responce.

### Image Processing

------------

[![](https://s3.amazonaws.com/sportsseam-public-read/NFL/demo/Capture.PNG)](https://s3.amazonaws.com/sportsseam-public-read/NFL/demo/Capture.PNG)



------------


[![](https://s3.amazonaws.com/sportsseam-public-read/NFL/demo/bin.png)](https://s3.amazonaws.com/sportsseam-public-read/NFL/demo/bin.png)

------------

## Architecture
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
