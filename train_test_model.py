import numpy as np
import os
import zipfile
import kaggle
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sn; sn.set(font_scale=1.4)
from sklearn.utils import shuffle           
import matplotlib.pyplot as plt             
import cv2                                 
import tensorflow as tf                
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")  

#Download dataset from kaggle 
# Create data directory & Info about Classes & Images

data_dir = "D:\\DL_Project\\data"

if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# Download from Kaggle (requires kaggle.json configured)
print("Downloading dataset from Kaggle...")
os.system('kaggle datasets download -d meowmeowmeowmeowmeow/gtsrb-german-traffic-sign')

# Extract
zip_path = "gtsrb-german-traffic-sign.zip"
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(data_dir)

print("✅ Dataset extracted successfully to:", data_dir)


# Load the training CSV file
train_csv = pd.read_csv(os.path.join(data_dir, "Train.csv"))
print(train_csv)
'''
test_path = os.path.join(data_dir, train_csv.iloc[0]["Path"]).replace('/', os.sep).replace('\\', os.sep)
print("Checking first image path:", test_path)
print("Exists:", os.path.exists(test_path))
'''
# Number of unique classes (should be 43)
nb_classes = train_csv["ClassId"].nunique()
print("Number of classes:", nb_classes)

classes = {
    0:'Speed limit (20km/h)', 1:'Speed limit (30km/h)', 2:'Speed limit (50km/h)',
    3:'Speed limit (60km/h)', 4:'Speed limit (70km/h)', 5:'Speed limit (80km/h)',
    6:'End of speed limit (80km/h)', 7:'Speed limit (100km/h)', 8:'Speed limit (120km/h)',
    9:'No passing', 10:'No passing veh over 3.5 tons', 11:'Right-of-way at intersection',
    12:'Priority road', 13:'Yield', 14:'Stop', 15:'No vehicles', 16:'Veh > 3.5 tons prohibited',
    17:'No entry', 18:'General caution', 19:'Dangerous curve left', 20:'Dangerous curve right',
    21:'Double curve', 22:'Bumpy road', 23:'Slippery road', 24:'Road narrows on the right',
    25:'Road work', 26:'Traffic signals', 27:'Pedestrians', 28:'Children crossing',
    29:'Bicycles crossing', 30:'Beware of ice/snow', 31:'Wild animals crossing',
    32:'End speed + passing limits', 33:'Turn right ahead', 34:'Turn left ahead',
    35:'Ahead only', 36:'Go straight or right', 37:'Go straight or left', 38:'Keep right',
    39:'Keep left', 40:'Roundabout mandatory', 41:'End of no passing', 42:'End no passing veh > 3.5 tons'
}

# Create a mapping: classid -> class name
class_names_label = classes

# Example lookup
print("Class 2 means:", class_names_label[2])

# Load Data

IMAGE_SIZE = (32, 32)
def load_data(df, base_dir):
    images = []
    labels = []

    for i, row in tqdm(df.iterrows(), total=len(df)):
        # Ensure path is constructed safely and works on any OS
        img_path = os.path.join(base_dir, row["Path"].replace('/', os.sep).replace('\\', os.sep))
        print(img_path)
        label = int(row["ClassId"])

        # Check if the image file exists
        if not os.path.exists(img_path):
            print(f"[Warning] Missing file: {img_path}")
            continue

        # Try reading image
        image = cv2.imread(img_path)
        if image is None:
            print(f"[Warning] Failed to load image: {img_path}")
            continue

        # Process
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, IMAGE_SIZE)
        image = image.astype("float32") / 255.0  # Normalize

        images.append(image)
        labels.append(label)

    if len(images) == 0:
        raise ValueError("No images loaded. Please verify dataset paths.")

    return np.array(images, dtype="float32"), np.array(labels, dtype="int32")


# load
test_csv  = pd.read_csv(os.path.join(data_dir, "Test.csv"))

train_images, train_labels = load_data(train_csv, data_dir)
test_images, test_labels = load_data(test_csv, data_dir)

train_images, train_labels = shuffle(train_images, train_labels, random_state=25)

n_train = train_labels.shape[0]
n_test = test_labels.shape[0]

print("Number of training examples:", n_train)
print("Number of testing examples:", n_test)
print("Each image is of size:", IMAGE_SIZE)

train_ids, train_counts = np.unique(train_labels, return_counts=True)
test_ids, test_counts = np.unique(test_labels, return_counts=True)

df = pd.DataFrame({
    'Train': pd.Series(train_counts, index=train_ids),
    'Test': pd.Series(test_counts, index=test_ids)
})

# Add class names
df.index = [classes[i] for i in df.index]

# Plot
ax = df.plot.bar(figsize=(18,6), width=0.8, alpha=0.85, color=["#1f77b4", "#ff7f0e"])

plt.title("Train vs Test Samples per Class", fontsize=16, fontweight="bold")
plt.ylabel("Number of Images", fontsize=13)
plt.xlabel("Traffic Sign Classes", fontsize=13)
plt.xticks(rotation=90, fontsize=9)   # Rotate x labels for readability
plt.yticks(fontsize=10)

plt.legend(title="Dataset", fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()


def display_random_image(classes, images, labels):
    """
        Display a random image from the images array and its corresponding label.
    """
    index = np.random.randint(images.shape[0])
    
    plt.figure(figsize=(3,3))
    plt.imshow(images[index])   
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.title(f'Image #{index} : {classes[labels[index]]}', fontsize=12)
    plt.show()
    
display_random = display_random_image(classes, train_images, train_labels)
print(display_random)
 
def display_examples(class_names, images, labels):
    """
    Display 25 images from the images array with its corresponding labels
    """
    
    fig = plt.figure(figsize=(20,20))
    fig.suptitle("Some examples of images of the dataset", fontsize=16)
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i])  # RGB image, no cmap
        plt.xlabel(classes[labels[i]])
        plt.show()
    
display = display_examples(classes, train_images, train_labels)
print(display)

# CNN Model

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout, BatchNormalization



# Building the model
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu', input_shape=train_images.shape[1:]))
model.add(Conv2D(filters=64, kernel_size=(5,5), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.15))
model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(filters=256, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.20))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(43, activation='softmax'))

# Compilation of the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#Model display
model.summary() 

# Training the Model
with tf.device('/GPU:0'):
    epochs = 32
    history1 = model.fit(train_images, train_labels, batch_size=16, epochs=epochs, validation_data=(test_images, test_labels))

model.save("D:\\DL_Project\\model\\traffic_sign_model.h5")


# Plot Accuracy

def plot_accuracy_loss(history):
    
    """
    Plot the accuracy and the loss during the training of the nn.
    """
    
    fig = plt.figure(figsize=(12,5))

    # Plot accuracy
    plt.subplot(1,2,1)
    plt.plot(history1.history['accuracy'], 'bo--', label='train_accuracy')
    plt.plot(history1.history['val_accuracy'], 'ro--', label='val_accuracy')
    plt.title("Train vs Validation Accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()

    # Plot loss function
    plt.subplot(1,2,2)
    plt.plot(history1.history['loss'], 'bo--', label='train_loss')
    plt.plot(history1.history['val_loss'], 'ro--', label='val_loss')
    plt.title("Train vs Validation Loss")
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.legend()

    plt.show()

plot_accuracy_loss(history1)

# Evaluate model performance on test data
print("\nEvaluating model on test data...")
test_loss, test_accuracy = model.evaluate(test_images, test_labels, verbose=2)

print(f"\nTest Accuracy: {test_accuracy * 100:.2f}%")
print(f"Test Loss: {test_loss:.4f}")

# Plot accuracy and loss for train, validation, and test sets
def plot_accuracy_loss_extended(history, test_loss, test_accuracy):
    epochs_range = range(1, len(history.history['accuracy']) + 1)

    plt.figure(figsize=(14, 6))

    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history.history['accuracy'], 'bo--', label='Train Accuracy')
    plt.plot(epochs_range, history.history['val_accuracy'], 'ro--', label='Validation Accuracy')
    plt.axhline(y=test_accuracy, color='green', linestyle='--', label='Test Accuracy')
    plt.title("Test Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history.history['loss'], 'bo--', label='Train Loss')
    plt.plot(epochs_range, history.history['val_loss'], 'ro--', label='Validation Loss')
    plt.axhline(y=test_loss, color='green', linestyle='--', label='Test Loss')
    plt.title("Test Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()

# Call the extended plotting function
plot_accuracy_loss_extended(history1, test_loss, test_accuracy)

#Predictions

from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns


pred_probs = model.predict(test_images)         
pred_labels = np.argmax(pred_probs, axis=1)

classes = [str(i) for i in range(43)]

print("Classification Report:")
print(classification_report(test_labels, pred_labels, target_names=classes))

#confusion matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(test_labels, pred_labels)

for i, cls in enumerate(classes):
    plt.figure(figsize=(3.5,3.5))
    
    # بناء مصفوفة 2x2 للكلاس
    cm_class = [[cm[i,i], cm[i].sum() - cm[i,i]],
                [cm[:,i].sum() - cm[i,i], cm.sum() - cm[i].sum() - cm[:,i].sum() + cm[i,i]]]
    
    # رسم
    ax = sns.heatmap(cm_class, annot=True, fmt="d", cmap="viridis", cbar=False,
                     xticklabels=[f"Pred {cls}", "Pred Others"],
                     yticklabels=[f"True {cls}", "True Others"],
                     annot_kws={"size":16, "weight":"bold", "color":"white"})  # أرقام كبيرة وBold
    
    plt.title(f"Confusion for class: {cls}", fontsize=16, weight="bold", color="darkblue")
    plt.xticks(fontsize=12, weight="bold", color="black")
    plt.yticks(fontsize=12, weight="bold", color="black", rotation=0)
    plt.tight_layout()
    plt.show()