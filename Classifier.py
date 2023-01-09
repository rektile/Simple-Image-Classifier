from classes.ImageClassifier import ImageClassifier
import argparse

parser = argparse.ArgumentParser(prog="Simple Image Classifier",
                                 description="Program to train ML algorithms to classify images. "
                                             "It uses data from the pictures folder. "
                                             "The folder name counts as the label/target.")


parser.add_argument("-m",
                    "--model",
                    help="ML model that will be used for training.",
                    choices=["SVC", "RandomForest"])

parser.add_argument("-v",
                    "--verbose",
                    help="Shows more info messages.",
                    action='store_true')

parser.add_argument("-r",
                    "--resolution",
                    help="Resolution the image will get resized to. Default=(150, 150)",
                    nargs="?",
                    default=(150, 150))

parser.add_argument("-s",
                    "--save",
                    help="The name you want to give to the saved model. This will be saved in the models folder.",
                    nargs="?")

parser.add_argument("-l",
                    "--load",
                    help="The name of the model you want to load. This will be loaded from the models folder.",
                    nargs="?")

parser.add_argument("-p",
                    "--predict",
                    help="The name of the picture you want the model to predict. This will be loaded from the predict folder.",
                    nargs="+")

args = parser.parse_args()

classifier = ImageClassifier()
classifier.argumentParser(args)
classifier.run()
