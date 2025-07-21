import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.activations import relu, linear, sigmoid

#Load data from MNIST dataset 
(X_train,y_train),(X_test,y_test) = tf.keras.datasets.mnist.load_data()

#Normalize the data
X_train,X_test = X_train/255.0, X_test/255.0

#reshape the data
X_train = X_train.reshape(-1,28*28)
X_test = X_test.reshape(-1,28*28)
print(X_train.shape)
print(X_test.shape)


#Define model
tf.random.set_seed(1234)
model=Sequential([
    tf.keras.Input(shape=(784,)),
    Dense(units=25, activation='relu', name = 'L1'),
    Dense(units=15, activation='relu', name = 'L2'),
    Dense(units=10, activation='linear', name = 'L3')
], name = "my_model")
model.summary()

#Compile model
model.compile(
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), #Define loss 
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001), #Define optimizer
    metrics=['accuracy']
)

#Fit the model to the training sets
history=model.fit(
    X_train,
    y_train, 
    epochs=10)

#prediction
prediction = model.predict(X_test[0].reshape(1,784))
prediction_p = tf.nn.softmax(prediction)
yhat = np.argmax(prediction_p)
print(yhat)

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Assume images are 20x20 pixels in dataset X
m, n = X_test.shape
y = y_test
X = X_test
# Set up plot
fig, axes = plt.subplots(8, 8, figsize=(5, 5))
fig.tight_layout(pad=0.13, rect=[0, 0.03, 1, 0.91])  # [left, bottom, right, top]

# Loop through each subplot and display random images with predictions
for i, ax in enumerate(axes.flat):
    # Select random indices
    random_index = np.random.randint(m)
    
    # Select and reshape the random image for display
    X_random_reshaped = X[random_index].reshape((28, 28)).T  # Transpose for correct orientation
    ax.imshow(X_random_reshaped, cmap='gray')
    
    # Predict using the model (reshape to match model's expected input shape)
    prediction = model.predict(X[random_index].reshape(1, 784))
    prediction_p = tf.nn.softmax(prediction)
    yhat = np.argmax(prediction_p)
    
    # Display the label above the image
    ax.set_title(f"{y[random_index]},{yhat}", fontsize=10)
    ax.set_axis_off()

fig.suptitle("Label, yhat", fontsize=14)
plt.show()

def display_errors(model, X, y):
    # Make predictions on the entire dataset
    predictions = model.predict(X, verbose=0)
    predictions_p = tf.nn.softmax(predictions)
    y_pred = np.argmax(predictions_p, axis=1)

    # Count the number of errors
    errors = np.sum(y_pred != y)
    
    return errors

# Call the function and display the result
print(f"{display_errors(model, X, y)} errors out of {len(X)} images")

