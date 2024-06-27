# Fruit-classifier Model

This project demonstrates a fruit recognition model built using TensorFlow and Keras. The model is capable of classifying images of various fruits into their respective categories. Below is a step-by-step guide to understanding and running this project.

## Table of Contents
1. [Installation](#installation)
2. [Dataset](#dataset)
3. [Model Architecture](#model-architecture)
4. [Training](#training)
5. [Evaluation](#evaluation)
6. [Usage](#usage)
7. [Results](#results)

## Installation

To get started, clone this repository and install the necessary dependencies:

```bash
git clone https://github.com/yourusername/fruit-recognition-model.git
cd fruit-recognition-model
pip install -r requirements.txt
```

## Dataset

The dataset used for this project contains images of different fruits organized in separate folders. Ensure that your dataset directory is structured as follows:

```
dataset/
    ├── apple/
    │   ├── image1.jpg
    │   ├── image2.jpg
    └── banana/
        ├── image1.jpg
        ├── image2.jpg
    ...
```

Update the `base_dir` variable in the code to point to your dataset directory.

## Model Architecture

The model is a Convolutional Neural Network (CNN) built using TensorFlow and Keras. It includes several convolutional layers, batch normalization, pooling layers, and dropout for regularization. Data augmentation is applied to improve the model's generalization.

```python
model = Sequential([
    data_augmentation,
    layers.Rescaling(1./255),
    Conv2D(32, 3, padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D(),
    Conv2D(128, 3, padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D(),
    Dropout(0.3),
    Flatten(),
    Dense(1024, activation='relu'),
    Dropout(0.3),
    Dense(512, activation='relu'),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')
])
```

## Training

To train the model, run the following script:

```python
history = model.fit(train_ds, epochs=100, validation_data=val_ds)
```

The model will be trained for 100 epochs with a batch size of 32.

## Evaluation

After training, evaluate the model's performance using a confusion matrix and classification report:

```python
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Get the true labels from the validation dataset
true_labels = []
for images, labels in val_ds:
    true_labels.extend(labels.numpy())

# Get the predicted labels from the model
predicted_scores = model.predict(val_ds)
predicted_labels = np.argmax(predicted_scores, axis=1)

# Calculate confusion matrix
conf_matrix = confusion_matrix(true_labels, predicted_labels)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=fruits_names, yticklabels=fruits_names)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

# Calculate classification report
class_report = classification_report(true_labels, predicted_labels, target_names=fruits_names)
print("Classification Report:")
print(class_report)
```

## Usage

To classify a new image, use the `classify_images` function:

```python
def classify_images(image_path):
    input_image = tf.keras.utils.load_img(image_path, target_size=(180, 180))
    input_image_array = tf.keras.utils.img_to_array(input_image)
    input_image_exp_dim = tf.expand_dims(input_image_array, 0)
    predictions = model.predict(input_image_exp_dim)
    result = tf.nn.softmax(predictions[0])
    outcome = 'The Image belongs to ' + fruits_names[np.argmax(result)] + ' with a score of ' + str(np.max(result) * 100)
    return outcome

result = classify_images('/path/to/your/image.jpg')
print(result)
```

## Results

The model's performance can be visualized through the confusion matrix and classification report generated during the evaluation step. These tools help in understanding the accuracy and precision of the model for each fruit category.


