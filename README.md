# Handwritten_Digit_Identification
Handwritten Digit Identification using CNNS (Artificial Intelligence Uppsala University)

### From the project description:

MNIST ("Modified National Institute of Standards and Technology") is the de facto “hello world” dataset of computer vision. Since its release in 1999, this classic dataset of handwritten images has served as the basis for benchmarking classification algorithms. As new machine learning techniques emerge, MNIST remains a reliable resource for researchers and learners alike.

Aim:
In this competition, your goal is to correctly identify digits from a dataset of tens of thousands of handwritten images. You are expected to use deep neural networks in this task for the identification of the handwritten images. Note that the dataset you are provided here includes distorted images of handwritten digits, hence different from the original MNIST dataset.

Evaluation:
To receive the credits for this assignment, you need to use convolutional neural networks (CNN) and reach an accuracy above 91%. The evaluation metric for this competition is accuracy, i.e., the number of correctly identified handwritten digits over the total number of tested handwritten digits.

Hint:
If you are rusty in Python, please follow the crash course here.
If you would like to see an example on CNN and Keras, please see here.

### Data description:
Here we'll use the MNIST dataset which consists of greyscale handwritten digits. Our goal is to build a neural network that can take one of these images and predict the digit in the image.

The data files train.csv and test.csv contain gray-scale images of hand-drawn digits, from zero through nine.

Each image is 28 pixels in height and 28 pixels in width, for a total of 784 pixels in total. Each pixel has a single pixel-value associated with it, indicating the lightness or darkness of that pixel, with higher numbers meaning darker. This pixel-value is an integer between 0 and 255, inclusive.

The training data set, (train.csv), has 785 columns. The first column, called "label", is the digit that was drawn by the user. The rest of the columns contain the pixel-values of the associated image. The test data set, (test.csv), is the same as the training set, except that it does not contain the "label" column.

Each pixel column in the training set has a name like pixelx, where x is an integer between 0 and 783, inclusive. To locate this pixel on the image, suppose that we have decomposed x as x = i * 28 + j, where i and j are integers between 0 and 27, inclusive. Then pixelx is located on row i and column j of a 28 x 28 matrix, (indexing by zero).

For example, pixel31 indicates the pixel that is in the fourth column from the left, and the second row from the top, as in the ascii-diagram below.

File descriptions:
trainset.csv - the training set
testset.csv - the test set
samplesubmission.csv - a sample submission file in the correct format

trainset.csv:
column 1: label (which digit 0-9)
column 2-end: pixels

testset.csv:
column 1-end: pixels

samplesubmission.csv:
column 1: image id
column 2: label (which digit 0-9)
