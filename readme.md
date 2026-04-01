# MNIST Digit Classifier 

A Deep Learning project focused on the "Hello World" of computer vision: identifying handwritten digits using the MNIST dataset.

## Project Overview
The goal is to classify grayscale images ($28 \times 28$ pixels) of handwritten digits into their respective categories (0 through 9).

* **Dataset:** 70,000 images (60k train / 10k test).
* **Model:** Neural Network using ReLU activation and a Softmax output layer.
* **Target Accuracy:** >98%.

---

## Getting Started

### 1. Prerequisites
Ensure you have [Conda](https://docs.conda.io/en/latest/) or [Miniconda](https://docs.conda.io/en/miniconda.html) installed.

### 2. Environment Setup
Clone this repository and recreate the environment using the provided `environment.yml` file:

```bash
# Create the environment
conda env create -f environment.yml

# Activate the environment
conda activate mnist-env