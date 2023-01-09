from classes.PictureGatherer import PictureGatherer
import argparse

parser = argparse.ArgumentParser(prog="Simple TrainingData Scaper",
                                 description="Script to scrape training data from google images.")

parser.add_argument("-a",
                    "--amount",
                    help="Amount of images to be scraped. Default=150",
                    nargs="?",
                    type=int,
                    default=150)

parser.add_argument("-k",
                    "--keyword",
                    help="What keyword to search on.",
                    nargs="+",
                    required=True)

args = parser.parse_args()
gatherer = PictureGatherer(args.amount, args.keyword)
gatherer.run()

