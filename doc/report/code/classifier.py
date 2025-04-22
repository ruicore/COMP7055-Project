#===============ResEmoteNet===============

import kagglehub

# Download latest version
path = kagglehub.dataset_download("msambare/fer2013")

print("Path to dataset files:", path)

import shutil
import os

src = '/kaggle/input/fer2013'
dst = '/content/fer2013'

shutil.copytree(src, dst)

import os
import shutil
import random

train_dir = '/content/fer2013/train'
val_dir = '/content/fer2013/validation'

os.makedirs(val_dir, exist_ok=True)

for class_folder in os.listdir(train_dir):
    class_folder_path = os.path.join(train_dir, class_folder)

    os.makedirs(os.path.join(val_dir, class_folder), exist_ok=True)

    files = os.listdir(class_folder_path)

    num_val_files = int(0.1 * len(files))
    val_files = random.sample(files, num_val_files)

    for file in val_files:
        shutil.copy(os.path.join(class_folder_path, file), os.path.join(val_dir, class_folder, file))

    for file in val_files:
        os.remove(os.path.join(class_folder_path, file))


import os
import shutil

source_dir = '/content/fer2013'
destination_dir = '/content/OutDir'

folder_mapping = {'test': 'test',
                  'train': 'train',
                  'validation': 'val'}

for folder in ['test', 'train', 'validation']:
    folder_path = os.path.join(source_dir, folder)

    dest_subdir = os.path.join(destination_dir, folder_mapping[folder])
    os.makedirs(dest_subdir, exist_ok=True)

    for class_folder in os.listdir(folder_path):
        class_folder_path = os.path.join(folder_path, class_folder)

        dest_class_folder = os.path.join(dest_subdir, class_folder)
        os.makedirs(dest_class_folder, exist_ok=True)

        for index, image in enumerate(os.listdir(class_folder_path)):

            image_name, image_ext = os.path.splitext(image)
            new_image_name = f"{folder_mapping[folder]}_{index}_{class_folder}{image_ext}"
            shutil.move(os.path.join(class_folder_path, image), os.path.join(dest_class_folder, new_image_name))


import os
import shutil

def move_images(source_folder, destination_folder):
    # Create destination subfolders (train/test/val)
    for split in ['train', 'test', 'val']:
        os.makedirs(os.path.join(destination_folder, split), exist_ok=True)

    # Walk through source directory
    for root, dirs, files in os.walk(source_folder):
        for file in files:
            if file.endswith('.jpg') or file.endswith('.png'):
                source_path = os.path.join(root, file)

                # Determine which split this file belongs to based on filename
                if file.startswith('train_'):
                    dest_split = 'train'
                elif file.startswith('test_'):
                    dest_split = 'test'
                elif file.startswith('val_'):
                    dest_split = 'val'
                else:
                    continue  # skip files that don't match expected pattern

                # Move to appropriate subfolder
                destination_path = os.path.join(destination_folder, dest_split, file)
                shutil.move(source_path, destination_path)

# Replace with your paths
test_folder = '/content/OutDir'
destination_folder = '/content/OutDir2'

move_images(test_folder, destination_folder)


import os
import pandas as pd

path = '/content/OutDir2'

label_mapping = {
    "happy": 0,
    "surprise": 1,
    "sad": 2,
    "anger": 3,
    "disgust": 4,
    "fear": 5,
    "neutral": 6
}

for partition in ['train', 'test', 'val']:
    partition_path = os.path.join(path, partition)
    image_data = []

    for filename in os.listdir(partition_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):

            label_name = filename.split('_')[-1].split('.')[0]

            label_value = label_mapping.get(label_name.lower())

            if label_value is not None:
                image_data.append([filename, label_value])
            else:
                print(f"Warning: Unknown label '{label_name}' in file {filename}")

    df = pd.DataFrame(image_data, columns=["image_name", "class"])
    csv_file_path = os.path.join(path, f"{partition}_labels.csv")
    df.to_csv(csv_file_path, index=False)

    print(f"Created {csv_file_path} with {len(df)} entries")

print("All label files created successfully!")


! git clone https://github.com/ArnabKumarRoy02/ResEmoteNet.git

! pip install -r /content/ResEmoteNet/requirements.txt


!cp /content/ResEmoteNet/approach/ResEmoteNet.py /content/ResEmoteNet.py
!cp /content/ResEmoteNet/get_dataset.py /content/get_dataset.py

import torch
import pandas as pd
import numpy as np
from tqdm import tqdm

from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim
import matplotlib.pyplot as plt

from ResEmoteNet import ResEmoteNet
from get_dataset import Four4All

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

train_dataset = Four4All(csv_file='/content/OutDir2/train_labels.csv',
                         img_dir='/content/OutDir2/train', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
train_image, train_label = next(iter(train_loader))


val_dataset = Four4All(csv_file='/content/OutDir2/val_labels.csv',
                       img_dir='/content/OutDir2/val', transform=transform)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True)
val_image, val_label = next(iter(val_loader))


test_dataset = Four4All(csv_file='/content/OutDir2/test_labels.csv',
                        img_dir='/content/OutDir2/test', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
test_image, test_label = next(iter(test_loader))

print(f"Train batch: Image shape {train_image.shape}, Label shape {train_label.shape}")
print(f"Validation batch: Image shape {val_image.shape}, Label shape {val_label.shape}")
print(f"Test batch: Image shape {test_image.shape}, Label shape {test_label.shape}")

# Load the model
model = ResEmoteNet()


# Print the number of parameters
total_params = sum(p.numel() for p in model.parameters())
print(f'{total_params:,} total parameters.')

# Hyperparameters
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)

patience = 15
best_val_acc = 0
patience_counter = 0
epoch_counter = 0

num_epochs = 80

train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []
test_losses = []
test_accuracies = []

# Start training
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for data in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        inputs, labels = data[0], data[1]

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_acc = correct / total
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)

    model.eval()
    test_running_loss = 0.0
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data[0], data[1]
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

    test_loss = test_running_loss / len(test_loader)
    test_acc = test_correct / test_total
    test_losses.append(test_loss)
    test_accuracies.append(test_acc)

    model.eval()
    val_running_loss = 0.0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for data in val_loader:
            inputs, labels = data[0], data[1]
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_loss = val_running_loss / len(val_loader)
    val_acc = val_correct / val_total
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)

    print(f"Epoch {epoch+1}, Train Loss: {train_loss}, Train Accuracy: {train_acc}, Test Loss: {test_loss}, Test Accuracy: {test_acc}, Validation Loss: {val_loss}, Validation Accuracy: {val_acc}")
    epoch_counter += 1

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        patience_counter = 0
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        patience_counter += 1
        print(f"No improvement in validation accuracy for {patience_counter} epochs.")

    if patience_counter > patience:
        print("Stopping early due to lack of improvement in validation accuracy.")
        break

df = pd.DataFrame({
    'Epoch': range(1, epoch_counter+1),
    'Train Loss': train_losses,
    'Test Loss': test_losses,
    'Validation Loss': val_losses,
    'Train Accuracy': train_accuracies,
    'Test Accuracy': test_accuracies,
    'Validation Accuracy': val_accuracies
})
df.to_csv('result_four4all.csv', index=False)


#===============CNN===============

import os
import pandas as pd
import numpy as np
import kagglehub
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Download latest version
path = kagglehub.dataset_download("msambare/fer2013")
print("Path to dataset files:", path)

train = '/root/.cache/kagglehub/datasets/msambare/fer2013/versions/1/train'
test = '/root/.cache/kagglehub/datasets/msambare/fer2013/versions/1/test'

img_size=(48,48)
bth_size=32

# Augmenting Images for training set

trdatagen = ImageDataGenerator(
    rescale=1.0/255,
    validation_split=0.1
)

# Only Rescaling or Normalizing the pixel values for testing Set
tedatagen = ImageDataGenerator(rescale=1.0/255)

# Reading the training data from directory

traingen= trdatagen.flow_from_directory(
    train,
    target_size=img_size,
    batch_size=bth_size,
    color_mode='grayscale',
    class_mode='categorical',
    shuffle=True,
    subset='training')

# Reading the testing data from Directory

testgen = tedatagen.flow_from_directory(
    test,
    target_size=img_size,
    batch_size=bth_size,
    color_mode='grayscale',
    class_mode='categorical',
    shuffle=False
)

# Validation data which is a subset taken from the training data (10% of the training data)
valgen = trdatagen.flow_from_directory(
    train,
    target_size=img_size,
    batch_size=bth_size,
    color_mode='grayscale',
    class_mode='categorical',
    shuffle=True,
    subset='validation'
)

import keras as k
from keras.models import Sequential
from keras.layers import Conv2D,BatchNormalization,Activation,GlobalAveragePooling2D,Dropout,Dense,MaxPooling2D
from keras.callbacks import ReduceLROnPlateau, EarlyStopping

model = Sequential()

# CNN Layer 0
model.add(Conv2D(32, (3,3), padding='same', input_shape=(48, 48, 1)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

# CNN Layer 1
model.add(Conv2D(64, (3,3), padding='same'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

# CNN Layer 2
model.add(Conv2D(128, (5,5), padding='same'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

# CNN Layer 3
model.add(Conv2D(512, (3,3), padding='same'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

#CNN Layer 4
model.add(Conv2D(512, (3,3), padding='same'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

model.add(GlobalAveragePooling2D())

#Dense Layer 2
model.add(Dense(512))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

#Dense Layer 3
model.add(Dense(256))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

#Output Layer
model.add(Dense(7, activation='softmax'))

#Adam Optimizer
opt = k.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()


#Implementing Early Stopping and Learning Rate Reduction
es=EarlyStopping(monitor='val_loss', patience=50, verbose=2, restore_best_weights=True)
lr=ReduceLROnPlateau(monitor='val_loss', factor=0.0005, patience=10, verbose=1, min_delta=0.0001)

callbacks_list=[es,lr]

history = model.fit(
    traingen,
    epochs=50,
    validation_data=valgen,
    callbacks=callbacks_list
)

from sklearn.metrics import classification_report, confusion_matrix

print(f"Test Accuracy: {test_accuracy:.2f}")
y_pred = np.argmax(model.predict(testgen), axis=1)
y_true = testgen.classes
print("Classification Report:")
print(classification_report(y_true, y_pred))
conf_matrix = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(conf_matrix,'\n\n')
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d',
            xticklabels=list(label_map.values()),
            yticklabels=list(label_map.values()))
plt.title('Confusion Matrix', fontsize=18)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()
print("Training class indices:", traingen.class_indices)
print("Test class indices:", testgen.class_indices)




plt.figure(figsize=(12, 6))

# Plot Loss
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss', color='blue')
plt.plot(history.history['val_loss'], label='Validation Loss', color='red')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Plot Accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy', color='blue')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='red')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()




import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="white")  #  "darkgrid", "white", "ticks"

plt.figure(figsize=(12, 6))

# Plot Loss
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss', color='blue')
plt.plot(history.history['val_loss'], label='Validation Loss', color='red')
plt.title('Training and Validation Loss', fontsize=14)
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.legend()
plt.grid(True)

# Plot Accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy', color='blue')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='red')
plt.title('Training and Validation Accuracy', fontsize=14)
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()





from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

# Convert true labels to one-hot encoding for ROC calculations
y_true_onehot = label_binarize(y_true, classes=[0, 1, 2, 3, 4,5,6])
y_pred_prob = model.predict(testgen)

plt.figure(figsize=(10, 8))

# Plot ROC for each class
for i, class_name in label_map.items():
    fpr, tpr, _ = roc_curve(y_true_onehot[:, i], y_pred_prob[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{class_name} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.grid()
plt.show()


