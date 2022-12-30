
# countCarNBicycleInPdf

A quick and **Really Dirty** python script to search and count of car and bicycle in texte and image of a bunch of pdf file.
It's aim for a french audience (we search for voiture and vélo)

## Installation

Fetch the scripts

```bash
  git clone git@github.com:upCleat/countCarNBicycleInPdf.git
  cd countCarNBicycleInPdf
```

Install countCarNBicycleInPdf requirement with python

```bash
  pip3 install -r requirements.txt
```

with osX I have to add python-tk

```bash
  brew install python-tk
```

Fetch a pre-trained model like [here](https://imageai.readthedocs.io/en/latest/detection/) and
[here](https://github.com/OlafenwaMoses/ImageAI)

## Usage/Examples

Modify the pdf source in pdflist.txt

```bash
vi pdflist.txt
```

Run the script

```bash
python3 main.py
```

See the result

```bash
cat result.csv
```

## Contributing

Contributions are always welcome!

See `contributing.md` for ways to get started.

Please adhere to this project's `code of conduct`.

## Authors

- [@tisseurdetoile](https://github.com/tisseurdetoile/)
