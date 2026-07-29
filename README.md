# 🐱🐶 Cat vs Dog Image Classification using CNN

## Project Overview
This project classifies an input image as either a **Cat** or a **Dog** using a Convolutional Neural Network (CNN) built with TensorFlow and Keras.

---

## Technologies Used

- Python 3.x
- TensorFlow / Keras
- NumPy

---

## Project Structure

```
Cat_Dog_Classifier/
│
├── dataset/
│   ├── cat/
│   │   ├── cat1.jpg
│   │   ├── cat2.jpg
│   │   └── ...
│   │
│   └── dog/
│       ├── dog1.jpg
│       ├── dog2.jpg
│       └── ...
│
├── train.py
├── predict.py
├── cat_dog_model.keras
├── test.jpg
└── README.md
```

---

## Dataset

The dataset contains two folders:

- **cat** → All cat images
- **dog** → All dog images

> TensorFlow automatically uses the **folder name** as the class label.
> Image filenames do not matter.

Example:

```
dataset/
    cat/
        1.jpg
        abc.jpg
        image10.png

    dog/
        dog1.jpg
        xyz.jpg
        puppy.png
```

---

## Install Dependencies

```bash
pip install tensorflow numpy
```

---

## Train the Model

```bash
python train.py
```

After training, the model is saved as:

```
cat_dog_model.keras
```

---

## Predict an Image

```bash
python predict.py
```

Example Output:

```
Dog
```

or

```
Cat
```

---

## CNN Architecture

- Rescaling
- Conv2D (32 Filters)
- MaxPooling2D
- Conv2D (64 Filters)
- MaxPooling2D
- Flatten
- Dense (128)
- Dense (1 - Sigmoid)

---

## Loss Function

Binary Crossentropy

## Optimizer

Adam

## Epochs

100

---

## Notes

- Place all cat images inside the `cat` folder.
- Place all dog images inside the `dog` folder.
- Image names can be anything.
- Folder names determine the class labels.

---

## Future Improvements

- Increase dataset size.
- Add Data Augmentation.
- Use Transfer Learning (MobileNetV2, ResNet50).
- Deploy using Flask.

---

## Author

Amarjeet