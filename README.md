# mlrf-sciag-2024
Authors: Quentin Fournel & Elouan Vincent

# Usage
## Install dependencies
```bash
pip install -r requirements.txt
```

## Run the program
```bash
python main.py
```

## How to use

`main.py` is an example of how to use the our module.
All the usable functions are in `src/`.

`src/data/make_dataset.py` is used to generate the dataset and unpack it.

`src/models/*` contains implementation of our models using `sklearn`.

`src/features/*` contains implementation of our features extractor.

Our implementation of our models include a method `fit` to train the model 
and `test` to compute accuracy, confusion matrix and ROC curve on the test set.

# Data
You can download the dataset [here](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz).

# Report
In `report/` you can find our report in LaTeX and PDF format.
