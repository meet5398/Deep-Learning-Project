# Deep-Learning-Project

## Project 01 : Number Classification using Artificial Neural Network on Number Datasets 

### **Description**

This Python script demonstrates the process of building an ANN model for number classification. It uses the MNIST dataset, which consists of images of handwritten digits ranging from 0 to 9.

The script begins by loading the dataset, splitting it into training and testing sets. The images are then preprocessed by scaling their values between 0 and 1. Next, a sequential model is created with an input layer of 784 neurons (representing the flattened images) and an output layer of 10 neurons (representing the possible digit classes). The model is trained using the Adam optimizer and sparse categorical cross-entropy loss. After training, the model's accuracy is evaluated on the testing set.

To improve the model's performance, a hidden layer with 100 neurons and a ReLU activation function is added. The training and evaluation process is repeated, and the results are compared.

### **Prerequisites**

**Before running the script, ensure you have the following dependencies installed:** <br>

- TensorFlow
- NumPy
- Matplotlib
- Scikit-learn

You can install TensorFlow by running the following command:
* !pip install tensorflow

### **Usage**

1. Make sure you have the prerequisite packages installed.
2. Run the script using a Python interpreter.
3. The script will automatically download the MNIST dataset, preprocess it, train the model, and evaluate its performance.
4. The classification results will be displayed in a confusion matrix.

### **Results**

The model's accuracy and confusion matrix are shown in the output. The confusion matrix provides insights into the model's performance by comparing the predicted labels with the true labels of the test set.

### **Resources**

- [MNIST Dataset](https://keras.io/api/datasets/mnist/)
- [TensorFlow Documentation](https://www.tensorflow.org/api_docs)
