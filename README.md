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

## Project:02 - Home Prices Prediction using Gradient Descent

#### This project aims to predict home prices in Bangalore using two gradient descent algorithms: batch gradient descent and stochastic gradient descent. 
#### The code takes into account the area and number of bedrooms as features to predict the home prices.

### Dependencies
- numpy
- pandas
- matplotlib

### Running the Code
1. Ensure that you have the required dependencies installed.
2. Download the "homeprices_banglore.csv" dataset and place it in the same directory as the code.
3. Run the code in a Python environment.

### Code Explanation
The code is divided into several sections, each performing a specific task. Here's a breakdown of the major sections:

1. Data Preprocessing:
   - Scaling: The area and number of bedrooms features are scaled using the MinMaxScaler from sklearn.preprocessing.

2. Batch Gradient Descent:
   - The batch_gradient_descent function performs batch gradient descent to optimize the weights and biases.
   - It takes the scaled features and target values as inputs, along with the number of epochs and learning rate.
   - The function iteratively updates the weights and biases based on the calculated gradients and the cost function.
   - The cost and epoch lists are populated during the training process.
   - The function returns the optimized weights, biases, final cost, and the lists of costs and epochs.

3. Prediction:
   - The predict function takes the area, number of bedrooms, weights, and biases as inputs.
   - It scales the input values and predicts the home price using the trained weights and biases.
   - The function returns the predicted price in the original scale.

4. Stochastic Gradient Descent:
   - The stochastic_gradient_descent function implements stochastic gradient descent.
   - It randomly selects a sample from the dataset and updates the weights and biases based on that sample.
   - The cost and epoch lists are populated during the training process.
   - The function returns the optimized weights, biases, final cost, and the lists of costs and epochs.

5. Plotting:
   - The code includes plotting functions to visualize the cost vs. epochs for both batch and stochastic gradient descent.

### Usage Example
After running the code, you can use the predict function to make predictions. Here's an example:

predicted_price = predict(2600, 4, w, b)
print("Predicted Price:", predicted_price)


### Additional Information
- The code includes comments to explain each section and its functionality.
- Batch gradient descent considers the entire dataset for each update, while stochastic gradient descent randomly selects a sample for each update.
- The cost vs. epochs plots show the decreasing trend of the cost function as the models learn.

### References
- [NumPy Documentation](https://numpy.org/doc/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Matplotlib Documentation](https://matplotlib.org/stable/contents.html)

