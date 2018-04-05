# Cilia Segmentation

This repository contains several algorithms for cilia segmentation and other support scripts for further experiments. 

# Data

Data are sequences of frame images. There are 325 videos in total each with 100 consecutive frames of a grayscale video of cilia. 211 of them are for training and 114 for testing. 

`data img`

Each training video also has one mask image with 3 labels. Background pixels are labeled 0, cells are labeled 1 and cilia are labeled as 2. Picture above shows the data.

The output is one mask for each testing video with pixels of cilia being labeled as 2.

# Getting Started

Below are instructions for installing and using this package.

## Prerequisites

- [Anaconda](https://www.anaconda.com/)
- [Python 3.6](https://www.python.org/downloads/release/python-360/)


## Environment Setup


## Running the Tests

- Testing one module
```
python -m pytest tests/[file you want to test]
```
- Testing one function under one module
```
python -m pytest tests/[file you want to test]::[function name]
```

# Evaluation

Evaluation of the output masks is done by checking IoU. An illustration for calculating IoU is show asa below.

`img IoU`

## Test results

Algorithm | Parameter value | IoU |
----------|-----------------|-----|
Optical Flow | threshold = 110 |16.02|
Optical Flow | threshold = 100 |17.73|
Optical Flow | threshold = 90  |18.97|
Optical Flow | threshold = 80  |19.48|
Optical Flow | threshold = 75  |19.40|
Optical Flow | threshold = 70  |19.00|

# Authors

(Ordered alphabetically)

- **Ankit Vaghela** - [ankit-vaghela30](https://github.com/ankit-vaghela30)
- **Vyom Shrivastava** - [vyom1911](https://github.com/vyom1911)
- **Weiwen Xu** - [WeiwenXu21](https://github.com/WeiwenXu21)

See CONTRIBUTORS file for more details.

# License

This project is licensed under the MIT License - see the LICENSE file for details
