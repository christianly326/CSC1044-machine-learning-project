from win10toast import ToastNotifier
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNet
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
toaster = ToastNotifier()
# toaster.show_toast("Test",
#                    duration=10)



# Load the dataframes
train_df = pd.read_csv('dataset/Training_set.csv')
test_df = pd.read_csv('dataset/Testing_set.csv')

# Split training data
train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)

# Data generators
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    directory='dataset/train',
    x_col='filename',
    y_col='label',
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)

val_generator = train_datagen.flow_from_dataframe(
    dataframe=val_df,
    directory='dataset/train',
    x_col='filename',
    y_col='label',
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    directory='dataset/test',
    x_col='filename',
    y_col=None,
    target_size=(150, 150),
    batch_size=32,
    class_mode=None,
    shuffle=False
)

print(f"Number of training images: {len(train_df)}")
print(f"Number of validation images: {len(val_df)}")
print(f"Number of test images: {len(test_df)}")


pipe = Pipeline([
    ('classifier', LogisticRegression(max_iter=1000))
])

# Fit the pipeline
pipe.fit(X_train, np.argmax(y_train, axis=1))

# Evaluate the pipeline
accuracy = pipe.score(X_val, np.argmax(y_val, axis=1))

pipe.predict(X_val)
print(f"Validation Accuracy: {accuracy:.4f}")

def extract_features(generator, model, has_label=True):
    features = []
    labels = [] if has_label else None
    for X_batch, *y_batch in generator:
        features_batch = model.predict(X_batch)  # Extract features
        features.append(features_batch)
        if has_label:
            labels.append(y_batch)
        if len(features) * generator.batch_size >= generator.samples:
            break
    if has_label:
        return np.vstack(features), np.vstack(labels)
    else:
        return np.vstack(features)
