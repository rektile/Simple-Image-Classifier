import requests
from bs4 import BeautifulSoup
import os

class PictureGatherer:
    def __init__(self, amountOfPictures, keywords):
        self.imageFolder = "..\pictures"
        self.imageUrl = "https://www.google.com/search?q={}&tbm=isch&start={}"
        self.imagesPerPage = 20
        self.amountOfPictures = amountOfPictures
        self.words = keywords

    def run(self):
        for word in self.words:
            self.prepareImages(word, self.amountOfPictures)
        print("[*] Done!")


    def downloadImages(self, word, urlArray):
        if not os.path.exists(self.imageFolder):
            os.mkdir(self.imageFolder)

        if not os.path.exists(f"{self.imageFolder}\\{word}"):
            os.mkdir(f"{self.imageFolder}\\{word}")

        print("[*] Downloading pictures")
        for i, url in enumerate(urlArray):
            r = requests.get(url)
            if r.status_code == 200:
                with open(f"{self.imageFolder}/{word}/{word}_{i}.jpg", "wb") as f:
                    f.write(r.content)
            print(f"[*] Downloaded picture {i+1}/{len(urlArray)}")

    def prepareImages(self,word, amount):

        print(f"[*] Finding images of \"{word}\"")

        currentTotalImages = 0
        imagesUrlArray = []
        while currentTotalImages < amount:

            amountToGet = min(amount - currentTotalImages, self.imagesPerPage)

            curUrl = self.imageUrl.format(word, currentTotalImages)
            currentUrls = self.getImagesUrls(curUrl, amountToGet)

            imagesUrlArray += currentUrls

            currentTotalImages += self.imagesPerPage

        self.downloadImages(word, imagesUrlArray)

    def getImagesUrls(self, url, amount=None):
        if not amount or amount > self.imagesPerPage:
            amount = self.imagesPerPage

        r = requests.get(url)
        html = r.text
        soup = BeautifulSoup(html, "html.parser")
        tableOfImages = soup.find("table", {"class": "GpQGbf"})
        try:
            imagesTags = tableOfImages.find_all("img")
        except:
            print(f"[!] Got to max amount of images found")
            return []
        allImageUrls = []

        for i in range(amount):
            img = imagesTags[i]
            allImageUrls.append(img.get("src"))

        return allImageUrls