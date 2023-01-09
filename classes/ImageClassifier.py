import os
import numpy as np
import pandas as pd
import pickle
from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# TODO implement more model types
# TODO implement Gridsearch for models

class ImageClassifier:

    def __init__(self):
        self.imageFolder = "pictures"
        self.predictFolder = "predict"
        self.modelFolder = "models"

        self.modelName = None
        self.model = None
        self.modelType = None
        self.useSavedModel = None
        self.shouldSave = None
        self.verbose = None
        self.shape = None
        self.toPredict = None

        self.dataFrame = None

    def argumentParser(self, args):

        if not args.model and not args.load:
            print("[!] You need to specify a model")
            exit()

        if args.model and args.load:
            print("[!] Can't train and load a model at the same time.")
            exit()

        if args.save and args.load:
            print("[!] Can't load and save at the same time.")
            exit()

        if args.save:
            self.modelName = args.save
            self.useSavedModel = False
            self.shouldSave = True
        elif args.load:
            self.modelName = args.load
            self.useSavedModel = True

        self.verbose = args.verbose
        self.toPredict = args.predict
        self.modelType = args.model

        try:
            self.shape = tuple(args.resolution)
        except Exception as E:
            print("[!] Resolution is the wrong format. Example=(150,150)")
            exit()

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
            label = path.split("\\")[1]
            preparedImage = self.prepareImage(path)
            images.append(preparedImage)
            targets.append(label)

        images = np.array(images)
        targets = np.array(targets)

        df = pd.DataFrame(images)
        df["target"] = targets

        self.dataFrame = df

    def trainModel(self, df_train, df_test):
        if self.modelType == "SVC":
            self.trainSVC(df_train, df_test)
        elif self.modelType == "RandomForest":
            self.trainRandomForestClassifier(df_train, df_test)
        else:
            print("[!] Problem with finding ML algorithm.")

    def trainRandomForestClassifier(self, df_train, df_test):
        x_train, y_train = df_train.drop(columns=["target"]), df_train["target"]
        x_test, y_test = df_test.drop(columns=["target"]), df_test["target"]

        print("[*] Training model with RandomForestClassifier")
        self.model = RandomForestClassifier(verbose=self.verbose, n_jobs=-1)
        self.model.fit(x_train, y_train)

        print(f"[-] Accuracy on train set is: {self.model.score(x_train, y_train)}")
        print(f"[-] Accuracy on test set is: {self.model.score(x_test, y_test)}")

    def trainSVC(self, df_train, df_test):
        x_train, y_train = df_train.drop(columns=["target"]), df_train["target"]
        x_test, y_test = df_test.drop(columns=["target"]), df_test["target"]

        print("[*] Training model with SVC")
        self.model = SVC(kernel="linear", probability=True, verbose=self.verbose)
        self.model.fit(x_train, y_train)

        print(f"[-] Accuracy on train set is: {self.model.score(x_train, y_train)}")
        print(f"[-] Accuracy on test set is: {self.model.score(x_test, y_test)}")


    def splitData(self, df):
        return train_test_split(df, shuffle=True, test_size=0.20)

    def predictImage(self, imagePath):
        image = self.prepareImage(imagePath)

        image = image.reshape(1, -1)

        print()
        print(f"[*] Prediction for {imagePath}")
        print(f"[-] Model thinks its a {self.model.predict(image)[0]}")
        print(f"[-] Probability is {self.model.predict_proba(image)}")

    def saveModel(self):

        modelPath = f"{self.modelFolder}\\{self.modelName}"
        print(f"[*] Saving model to {modelPath}")

        if not os.path.exists(self.modelFolder):
            os.mkdir(self.modelFolder)

        with open(modelPath, "wb") as f:
            pickle.dump(self.model, f)

    def loadModel(self):
        modelPath = f"{self.modelFolder}\\{self.modelName}"
        print(f"[*] Loading model from {modelPath}")

        if not os.path.exists(self.modelFolder):
            os.mkdir(self.modelFolder)

        if not os.path.exists(modelPath):
            print(f"[!] Failed to find model {modelPath}")
            exit()

        with open(modelPath, "rb") as f:
            self.model = pickle.load(f)
            self.model.verbose = self.verbose
        print(f"[*] Model loaded: {self.model}")


    def run(self):

        if not self.useSavedModel:
            paths = self.getAllImagePaths()
            self.getTrainingData(paths)
            df_train, df_test = self.splitData(self.dataFrame)

            self.trainModel(df_train, df_test)

            if self.shouldSave:
                self.saveModel()
        else:
            self.loadModel()

        if self.toPredict:
            for image in self.toPredict:
                self.predictImage(f"{self.predictFolder}\\{image}")