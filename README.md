# Simple Image Classifier
## Info
A simple program that lets you train a ML model for image classification.  
There is also a helper script that lets you scrape training data from google images.  
This script only works until **python 3.10.9**

## Installation
Clone the repository
```shell
git clone https://github.com/rektile/Simple-Image-Classifier.git
```

Go into the folder
```shell
cd ./Simple-Image-Classifier
```

Install python requirements
```shell
pip install -r requirements.txt
```

## Image Classifier
There are 3 important folders inside this project you need to beware of.

- models: This is where your models are going to be saved and loaded from.
- pictures: This is where the training data needs to be.
- predict: This is where you put the pictures of the images you want to model to predict.



Get help screen for arguments
```shell
python Classifier.py -h
```

Example commands
```shell
# Uses the model SVC to train data, save it as SVC_model and predict the cat.jpg image
python Classifier.py -m SVC -s SVC_model -p cat.jpg

# Use the saved model named SVC_model, verbose output and predict cat.jpg and dog.jpg
python Classifier.py -l SVC_model -v -p cat.jpg dog.jpg
```
### How are images labeled?
The label they get are based on what folder they are in.  
If you have train data of dogs, put them inside a folder named dog inside of pictures.  
For e.g:
```shell
pictures/
├─ dog/
│  ├─ dog_0.jpg
├─ cat/
│  ├─ cat_0.jpg
```

## Scrape script

Get help screen for arguments
```shell
python TrainingDataScraper.py -h
```

Example commands
```shell
# Get 500 images each of dogs and cats
python TrainingDataScraper.py -a 500 -k dog cat
```

### Warning
These images are scraped from the top results of google.  
Because of this there could be random images that don't fit the keyword or are bad quality overall.  
This makes the overall results of the model bad.

