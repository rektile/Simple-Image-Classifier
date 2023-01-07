import os
import numpy as np
import pandas as pd
import pickle
from PIL import Image
from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

class ImageClassifier:

    def __init__(self):
        self.imageFolder = "..\pictures"
        self.predictFolder = "..\predict"
        self.modelFolder = "..\models"

        self.modelName = "newModel"
        self.model = None
        self.useSavedModel = True

        self.shape = (150, 150)
        self.dataFrame = None


    def getAllImagePaths(self):
        paths = []
        dirs = [dir for dir in os.listdir(self.imageFolder) if os.path.isdir(f"{self.imageFolder}\\{dir}")]

        for dir in dirs:
            curDirPath = f"{self.imageFolder}\\{dir}\\"
            files = os.listdir(curDirPath)

            for file in files:
                if os.path.isfile(curDirPath + file):
                    paths.append(curDirPath + file)

        return paths

    def prepareImage(self, path):
        image = imread(path)
        imageResized = resize(image, (self.shape[0], self.shape[1], 3))
        return imageResized.flatten()

    def getTrainingData(self, imagePaths):

        print(f"[*] Preparing training data")

        images = []
        targets = []

        for path in imagePaths:
            label = path.split("\\")[2]
            preparedImage = self.prepareImage(path)
            images.append(preparedImage)
            targets.append(label)

        images = np.array(images)
        targets = np.array(targets)

        df = pd.DataFrame(images)
        df["target"] = targets

        self.dataFrame = df

    def trainRandomForestClassifier(self, df_train, df_test):
        x_train, y_train = df_train.drop(columns=["target"]), df_train["target"]
        x_test, y_test = df_test.drop(columns=["target"]), df_test["target"]


        print("[*] Training model with RandomForestClassifier")
        self.model = RandomForestClassifier()
        self.model.fit(x_train, y_train)

        print(f"accuracy on train set is: {self.model.score(x_train, y_train)}")
        print(f"accuracy on test set is: {self.model.score(x_test, y_test)}")

    def trainSVC(self, df_train, df_test):
        x_train, y_train = df_train.drop(columns=["target"]), df_train["target"]
        x_test, y_test = df_test.drop(columns=["target"]), df_test["target"]

        print("[*] Training model with SVC")
        self.model = SVC(kernel="linear", probability=True)
        self.model.fit(x_train, y_train)

        print(f"accuracy on train set is: {self.model.score(x_train, y_train)}")
        print(f"accuracy on test set is: {self.model.score(x_test, y_test)}")


    def splitData(self, df):
        return train_test_split(df, shuffle=True, test_size=0.20)

    def predictImage(self, imagePath):
        image = self.prepareImage(imagePath)
        image = image.reshape(1, -1)
        print(f"[*] Prediction for {imagePath}")
        print(f"[-] Model thinks its a {self.model.predict(image)[0]} ")
        print(f"[-] Probability is {self.model.predict_proba(image)}")

    def saveModel(self):

        modelPath = f"{self.modelFolder}\\{self.modelName}"
        print(f"[*] Saving model to {modelPath}")

        # TODO check if dir
        with open(modelPath, "wb") as f:
           pickle.dump(self.model, f)

    def loadModel(self):
        modelPath = f"{self.modelFolder}\\{self.modelName}"
        print(f"[*] Loading model from {modelPath}")

        # TODO check if dir and model exists
        with open(modelPath, "rb") as f:
            self.model = pickle.load(f)

    def run(self):

        if not self.useSavedModel:
            paths = self.getAllImagePaths()
            self.getTrainingData(paths)
            df_train, df_test = self.splitData(self.dataFrame)

            #self.trainSVC(df_train, df_test)
            self.trainRandomForestClassifier(df_train, df_test)
            self.saveModel()
        else:
            self.loadModel()

        self.predictImage(f"{self.predictFolder}\\cat.jpg")
        self.predictImage(f"{self.predictFolder}\\dog.jpg")
        self.predictImage(f"{self.predictFolder}\\longcat.jpg")



i = ImageClassifier()

i.run()
#i.getAllImagePaths()