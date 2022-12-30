import csv
import fitz
import hashlib
import os
import uuid

from collections import Counter
from io import BytesIO
from imageai.Detection import ObjectDetection
from urllib.request import urlopen
from unidecode import unidecode


def main():
    print(fitz.__doc__)
    execution_path = os.getcwd()
    dwlData = "./dwl/"
    detector = ObjectDetection()

    """
    Récuperation d'un modèle déjà entrainé le RetinaNet Lent mais précis
    https://imageai.readthedocs.io/en/latest/detection/
    """
    # detector.setModelTypeAsRetinaNet()
    # detector.setModelPath(
    #    os.path.join(execution_path, "retinanet_resnet50_fpn_coco-eeacb38b.pth")
    # )

    """
    Récuperation d'un modèle déjà entrainé le TinyYOLOv3 Rapide peux précis
    https://imageai.readthedocs.io/en/latest/detection/
    """
    detector.setModelTypeAsTinyYOLOv3()
    detector.setModelPath(os.path.join(execution_path, "tiny-yolov3.pt"))

    detector.loadModel()

    listPdf = []

    header = [
        "numero",
        "texte_velo",
        "image_velo",
        "texte_voiture",
        "image_voiture",
        "hash",
        "url",
    ]

    with open("pdflist.txt", "r") as ins:
        for line in ins:
            listPdf.append(line.rstrip())

    with open("result.csv", "w", newline="") as csvfile:
        csvWriter = csv.writer(
            csvfile, delimiter=",", quotechar="|", quoting=csv.QUOTE_MINIMAL
        )
        csvWriter.writerow(header)

        for url in listPdf:
            print("traite :", url)
            velo = 0
            voiture = 0
            imgVoiture = 0
            imgVelo = 0

            text = ""
            title = ""

            pdf_data = BytesIO(urlopen(url).read())
            with fitz.open(stream=pdf_data, filetype="pdf") as doc:
                title = doc.metadata.get("title")

                if not title:
                    title = str(uuid.uuid1())

                hash = hashlib.sha1(title.encode("utf-8")).hexdigest()

                for page in doc:
                    print("traite :", url, "textes")
                    text = unidecode(page.get_text().lower())
                    # print(text)
                    wordlist_freq = Counter(text.split())
                    occVelo = wordlist_freq.get("velo", 0) + wordlist_freq.get(
                        "velos", 0
                    )

                    velo += occVelo
                    voiture += wordlist_freq.get("voiture", 0) + wordlist_freq.get(
                        "voitures", 0
                    )

                    ## get image
                    img_list = page.get_images()
                    for img in img_list:

                        xref = img[0]
                        pix = fitz.Pixmap(doc, xref)
                        imageName = "%sf%sp%s-%s.jpg" % (
                            dwlData,
                            hash,
                            page.number,
                            xref,
                        )
                        imageNameTs = "%sf%sp%s-%s_ts.jpg" % (
                            dwlData,
                            hash,
                            page.number,
                            xref,
                        )

                        minSize = pix.width > 150 and pix.height > 150

                        if minSize and pix.colorspace.name in (
                            fitz.csGRAY.name,
                            fitz.csRGB.name,
                        ):
                            print(
                                "traite :",
                                url,
                                "image: ",
                                imageName,
                                "w:",
                                pix.width,
                                "h:",
                                pix.height,
                            )
                            pix.save(imageName)

                            custom = detector.CustomObjects(car=True, bicycle=True)
                            ## detection d'image
                            detections = detector.detectObjectsFromImage(
                                custom_objects=custom,
                                input_image=os.path.join(execution_path, imageName),
                                output_image_path=os.path.join(
                                    execution_path, imageNameTs
                                ),
                                minimum_percentage_probability=60,
                            )

                            detectedObject = set()
                            found = False
                            for eachObject in detections:
                                detectedObject.add(eachObject["name"])

                            if "car" in detectedObject:
                                imgVoiture += 1
                                found = True
                            if "bicycle" in detectedObject:
                                found = True
                                imgVelo += 1

                            # on supprime les fichiers
                            os.remove(imageName)
                            if not found and os.path.exists(imageNameTs):
                                os.remove(imageNameTs)

                    ## fin image

            result = [title, velo, imgVelo, voiture, imgVoiture, hash, url]

            csvWriter.writerow(result)


if __name__ == "__main__":
    main()
