import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

# Load data
train_df = pd.read_csv('dataset/Training_set.csv')
test_df = pd.read_csv('dataset/Testing_set.csv')

# Split training data
train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)

# label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
# # print("\nLabel Mapping:")
# # for k, v in label_mapping.items():
# #     print(f"{k}: {v}")

val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Create data generators
train_generator = train_datagen.flow_from_dataframe(
      dataframe=train_df,
      directory='dataset/train',
      x_col='filename',
      y_col='label',
      target_size=(150, 150),
      batch_size=32,
      class_mode='categorical'
)

val_generator = val_datagen.flow_from_dataframe(
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
      target_size=(150, 150),
      batch_size=32,
      class_mode=None,
      shuffle=False
)

# Pipeline setup remains similar but adapted for generators
pipeSVM = Pipeline([
      ('classifier', SVC(kernel='rbf', random_state=42))
])

# Grid search with generators
param_grid = {
      'classifier__C': [0.1, 1, 10],
      'classifier__gamma': ['scale', 'auto'],
      'classifier__kernel': ['rbf', 'linear']
}

print("\nStarting grid search...")
grid_search = GridSearchCV(
      pipeSVM, 
      param_grid, 
      cv=3,
      verbose=True, 
      scoring='accuracy', 
      n_jobs=-1
)

# Train and evaluate using generators
grid_search.fit(train_generator, val_generator)
predictions = grid_search.predict(test_generator)