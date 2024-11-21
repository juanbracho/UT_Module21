# UT_Module21
 
# Alphabet Soup Charity Donation Prediction

## Project Overview
The Alphabet Soup nonprofit organization seeks a machine learning solution to identify applicants who are likely to successfully use their funding. This project uses a deep learning model to create a binary classifier that predicts the success of funded ventures based on historical data.

## Files in Repository
- **Starter_Code.ipynb**: Jupyter Notebook for preprocessing data, creating, training, and evaluating the model.
- **AlphabetSoupCharity.h5**: Saved model file for the trained neural network.

## Steps Taken in the Analysis
1. **Data Preprocessing**
   - Dropped irrelevant columns (`EIN` and `NAME`).
   - Encoded categorical variables using one-hot encoding.
   - Binned low-frequency categories in the `APPLICATION_TYPE` and `CLASSIFICATION` columns into an "Other" category.
   - Scaled numerical features using `StandardScaler`.

2. **Model Creation**
   - Designed a neural network using TensorFlow and Keras:
     - Input layer with 80 neurons.
     - Two hidden layers with 80 and 30 neurons, both using ReLU activation.
     - Output layer with a sigmoid activation function for binary classification.
   - Compiled the model with the Adam optimizer and binary cross-entropy loss function.

3. **Training**
   - Trained the model for 100 epochs with 20% validation data.

4. **Evaluation**
   - Evaluated the model on the test set, achieving an accuracy of approximately 72%.

5. **Model Optimization**
   - Created additional models with different hyperparameters and architectures to improve accuracy, aiming for at least 80%.

## Results
- **Initial Model Accuracy**: 72%
- **Optimization Attempts**: Additional attempts to improve accuracy through:
  - Adjusting the number of neurons and layers.
  - Modifying activation functions.
  - Tuning the training process (e.g., epochs and validation split).

## Requirements
To run the project, you’ll need:
- Python 3.x
- TensorFlow
- Keras
- Pandas
- Scikit-learn
- Jupyter Notebook

