import tempfile
from pathlib import Path
from PIL import Image
import os

import streamlit as st
from streamlit_option_menu import option_menu

import tensorflow as tf
from keras import layers 
from keras import Sequential
from keras import preprocessing
from keras import callbacks


st.set_page_config("Image Recognition", layout='wide')

# Check if user specified more classes than what he provided
def empty_class(train_files, test_files, placeholders):

    # Initialize a flag to check if any class is empty
    is_empty = False

    # Iterate over the uploaded files for each class
    for idx, train_class in enumerate(train_files + test_files):

        # Display an error message below the uploader widget if no files are uploaded on submit
        if not train_class:
            placeholders[idx].error("There should be at least one file in each class.")
            is_empty = True
    
    return is_empty


def create_uploader(num_class, dataset_seg, placeholders, label_prefix):
    
    for i in range(num_class):

        # Create file uploader widgets for the specified number of classes
        train_class = st.file_uploader(
            label=f"{label_prefix} Class {i+1}",
            type=["jpg", "jpeg", "png"], # file types accepted by widget
            accept_multiple_files=True,
            key = f"{label_prefix}{i}" 
        )

        # Create a placeholder for displaying error messages if any
        error_msg_placeholder = st.empty()
        
        # Store the file uploader results and placeholders
        dataset_seg.append(train_class)
        placeholders.append(error_msg_placeholder)


@st.cache_resource()
def create_model(num_class):
    
    model = Sequential()

    model.add(layers.RandomFlip("horizontal_and_vertical"))
    model.add(layers.RandomRotation((-1, 1)))
    model.add(layers.RandomTranslation(height_factor=0.2, width_factor=0.2))
    model.add(layers.RandomZoom(height_factor=(-0.3, -0.2), width_factor=(-0.3, -0.2)))
    model.add(layers.RandomBrightness(0.2))

    model.add(layers.Convolution2D(filters=32, kernel_size=(3, 3), input_shape = (256, 256, 3), activation = 'relu'))
    model.add(layers.MaxPooling2D(pool_size = (2, 2)))
    model.add(layers.BatchNormalization())

    model.add(layers.Convolution2D(filters=64, kernel_size=(3, 3), activation = 'relu'))
    model.add(layers.MaxPooling2D(pool_size = (2, 2)))
    model.add(layers.BatchNormalization())

    model.add(layers.Convolution2D(filters=128, kernel_size=(3, 3), activation = 'relu'))
    model.add(layers.MaxPooling2D(pool_size = (2, 2)))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.25))

    model.add(layers.Flatten())

    model.add(layers.Dense(128, activation = 'relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.25))
    
    model.add(layers.Dense(num_class, activation = 'softmax'))
    
    return model


def fit_model(model, data_dir, training_set, test_set):

    model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

    early_stop = callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    #model_cp = callbacks.ModelCheckpoint(filepath=data_dir, monitor="val_loss", save_best_only=True)
    lr_rate = callbacks.ReduceLROnPlateau(monitor="val_loss", patience=2, factor=0.1)

    model.fit(
        training_set,
        steps_per_epoch=None,
        epochs=20,
        validation_data=test_set,
        validation_steps=None,
        callbacks=[early_stop, lr_rate]
    )



def split_data(data_dir, train_size):

    # Define directory and parameters
    data_dir = data_dir  # Replace with the path to your dataset
    batch_size = 32
    img_height = 256
    img_width = 256

    # Create training and validation datasets
    train_ds = preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=1-train_size,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        pad_to_aspect_ratio=True,
        batch_size=batch_size
    )

    val_ds = preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=1-train_size,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        pad_to_aspect_ratio=True,
        batch_size=batch_size
    )
    
    return train_ds, val_ds


def train_model(training_files):

    # Create temporary directory to house uploaded files
    # Temp directory will retain the the same directory structure as per uploaded files
    with tempfile.TemporaryDirectory() as tmpdirname:
        for class_idx, files in enumerate(training_files):
            class_dir = Path(tmpdirname) / f"class_{class_idx+1}"
            class_dir.mkdir(parents=True, exist_ok=True)
            
            for file in files:
                img = Image.open(file)
                img_path = class_dir / file.name
                img.save(img_path)

        train_ds, val_ds = split_data(tmpdirname, train_size)
        model=create_model(num_class)
        fit_model(model, tmpdirname, train_ds, val_ds)
        
        return 


with st.sidebar:
    nav = option_menu("Image Recognition", options=["Training", "Predictions"], menu_icon=["eye-fill"], icons=["tools", "bullseye"], orientation="vertical")

st.title("Image Recognition")
st.divider()

if nav == "Training":

    # Initialise lists to store uploaded files for each class and placeholders for error messages
    train_files = []
    test_files = []
    placeholders = []

    st.subheader("Training")

    with st.container(border=True):

        # Takes a user input for number of image classes
        num_class = st.number_input("Number of Classes", min_value=1, step=1)
        num_class = int(num_class)
        
        # Allows the user to choose how to split their dataset
        # Provides flexibility for the user in the file structure
        upload_option = st.selectbox("Upload options", options=["Auto Split", "Manual Upload"])


        if upload_option == "Auto Split": 

            with st.form("auto_split", border=False):

                # User specified train test split for dataset
                train_size = st.slider("Train Test Split", min_value=0.0, max_value=1.0, value=0.8)
                train_size = float(train_size)

                # Create file uploaders
                st.subheader("Upload files")
                create_uploader(num_class, train_files, placeholders, "")

                submit = st.form_submit_button("Train Model")
                
                if submit:
                    
                    # Proceed with model training since there are no empty classes
                    if not empty_class(train_files, test_files, placeholders):
                        train_model(train_files)
                
                    else:
                        pass

                
        else:

            with st.form("manual_split", border=False):

                # Create individual columns for uploading of training and testing data set
                train_uploader_col, test_uploader_col = st.columns(2, gap="large")

                # Create file uploaders
                with train_uploader_col:
                    
                    st.subheader("Training files")
                    create_uploader(num_class, train_files, placeholders, "Train")    
                    
                with test_uploader_col:
                    st.subheader("Testing files")
                    create_uploader(num_class, test_files, placeholders, "Test")

                submit = st.form_submit_button("Train Model")


        #if submit:
        #    # Proceed with model training since there are no empty classes
        #    if not empty_class(train_files, test_files, placeholders):
        #        train_ds, val_ds = split_data(train_files, train_size)
        #        print(train_ds)
        #        print(val_ds)
        #        
        #    else:
        #        pass

    
    
    

                
elif nav == "Predictions":
    st.subheader("Predict image class")
    st.file_uploader("Upload image to be predicted", type=["jpg", "png", "jpeg"])
    st.button("Predict Image")