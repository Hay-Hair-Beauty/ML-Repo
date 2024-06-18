# Hay Healthy - Scan Hair Predict
This is a Capstone Project part Machine Learning-- <br/>
We created a Tensorflow model to predict hair on a person. Thus, we hope to easily predict whether a user has a hair disease or not.

## Project Structure
```bash
├── datasets
│   ├── test "--> Our test datasets, contains with 12 labels"
│   │   ├── Alopecia Areata
│   │   ├── Contact Dermatitis
│   │   ├── Dry Hair
│   │   ├── Folliculitis
│   │   ├── Head Lice
│   │   ├── Healthy Hair
│   │   ├── Lichen Planus
│   │   ├── Male Pattern Baldness
│   │   ├── Psoriasis
│   │   ├── Seborrheic Dermatitis
│   │   ├── Telogen Effluvium
│   │   └── Tinea Capitis
│   ├── train "--> Our train datasets, contains with 12 labels"
│   │   ├── Alopecia Areata
│   │   ├── Contact Dermatitis
│   │   ├── Dry Hair
│   │   ├── Folliculitis
│   │   ├── Head Lice
│   │   ├── Healthy Hair
│   │   ├── Lichen Planus
│   │   ├── Male Pattern Baldness
│   │   ├── Psoriasis
│   │   ├── Seborrheic Dermatitis
│   │   ├── Telogen Effluvium
│   │   └── Tinea Capitis
│   └── val " --> Our val dataset, contains with 12 labels "
│       ├── Alopecia Areata
│       ├── Contact Dermatitis
│       ├── Dry Hair
│       ├── Folliculitis
│       ├── Head Lice
│       ├── Healthy Hair
│       ├── Lichen Planus
│       ├── Male Pattern Baldness
│       ├── Psoriasis
│       ├── Seborrheic Dermatitis
│       ├── Telogen Effluvium
│       └── Tinea Capitis
├── Test Images
│   ├── Prediksi.jpeg "--> An image that predicts a disease in the hair"
│   ├── Prediksi2.jpg "--> An image that predicts a disease in the hair"
│   └── Prediksi3.jpg "--> An image that predicts a disease in the hair"
├── .gitattributes "--> sending large files from tflite via Github Large File"
├── Hay-Models.h5 "--> a convert model to TFDH5 file for mobile"
├── HayModels.ipynb "--> a baseline notebook python"
├── HayModels.tflite "--> a convert model from TFDH5 to tflite file for mobile"
├── README.md "--> documentation our repository"
├── data_recomm.csv "--> a data set that recommends a shampoo/treatment"
├── label.txt "--> label of a hair diseases"
├── recommender_system.py "--> a file of recommender system"
└── tflite_converter.py "--> A file that converts tflite without metadata into tflite with metadata"
```

## Datasets
Training Data and Testing Data that were used are sourced from kaggle :https://www.kaggle.com/datasets/sundarannamalai/hair-diseases <br/> The dataset contains 12,082 images of apple, banana, and orange divided into fresh and rotten each.
Dataset       | Directories           | Files
------------- | -------------         | -------------
Test          | Alopecia Areata       | 120
|             | Contact Dermatitis    | 120
|             | Dry Hair              | 144
|             | Folliculitis          | 120
|             | Head Lice             | 120
|             | Healthy Hair          | 120
|             | Lichen Planus         | 120
|             | Male Pattern Baldness | 120
|             | Psoriasis             | 120
|             | Seborrheic Dermatitis | 120
|             | Telogen Effluvium     | 120
|             | Tinea Capitis         | 120
Train         | Alopecia Areata       | 1000
|             | Contact Dermatitis    | 1000
|             | Dry Hair              | 1000
|             | Folliculitis          | 1000
|             | Head Lice             | 1000
|             | Healthy Hair          | 1082
|             | Lichen Planus         | 1000
|             | Male Pattern Baldness | 1000
|             | Psoriasis             | 1000
|             | Seborrheic Dermatitis | 1000
|             | Telogen Effluvium     | 1000
|             | Tinea Capitis         | 1000
val           | Alopecia Areata       | 120
|             | Contact Dermatitis    | 120
|             | Dry Hair              | 120
|             | Folliculitis          | 120
|             | Head Lice             | 120
|             | Healthy Hair          | 120
|             | Lichen Planus         | 120
|             | Male Pattern Baldness | 120
|             | Psoriasis             | 120
|             | Seborrheic Dermatitis | 120
|             | Telogen Effluvium     | 120
|             | Tinea Capitis         | 120


## Network
For this model, we use Convolutional Neural Networks. Our model used transfer learning InceptionV3 for the baseline model with 12 classes.

## Prequisites
You don't need to install anything since its written in Google Colab which is a cloud service
- Download this repository first. And in .zip form
- After succesfully download and open extract zip, after that you open folder in local computer. Please change the storage directory of your notebook.
- If you are using Visual Studio Code, then first download the source code, such as hard tensorflow, tensorflow, pandas, matplotlib, and others. If you are using Google Collab, just run it.
- If you have everything, just run it.

## Built With
* [Tensorflow](https://www.tensrflow.org) - The AI framework used
* [Tensorflow Keras](https://www.tensrflow.org) - The AI framework used
* [Numpy](https://numpy.org/) and [Matplotlib](https://matplotlib.org/)- The Visualization
* [Pandas](https://pandas.pydata.org/)- The Open Sources Data Analyst

## TFLite Converter
To convert .h5 to tflite that uses metadata. There are several steps. As follows:
- Once you have confirmed that your model is perfect, then first save it to .h5. Like this:
```bash
model_for_export = tfmot.sparsity.keras.strip_pruning(model)
_, pruned_keras_file = tempfile.mkstemp('.h5')
tf.keras.models.save_model(model_for_export, pruned_keras_file, include_optimizer=False)
print('Saved pruned Keras model to:', pruned_keras_file)
```

- After that, run the code. The .h5 file is automatically created. The next step, change .h5 to tflite. Like this:
```bash
converter = tf.lite.TFLiteConverter.from_keras_model(model_for_export)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
quantized_and_pruned_tflite_model = converter.convert()

_, quantized_and_pruned_tflite_file = tempfile.mkstemp('.tflite')

with open(quantized_and_pruned_tflite_file, 'wb') as f:
  f.write(quantized_and_pruned_tflite_model)

print('Saved quantized and pruned TFLite model to:', quantized_and_pruned_tflite_file)
```

- If so, run it. Then as usual, the .h5 file is automatically formed into a tflite file. But the tflite cannot be used yet, because we need metadata so that it can be used in Android Studio.
- Before you open tflite_converter.py. First install tflite_support. 
```bash
pip install tflite_support
```
- You must have label.txt, the point is for tflite_converter.py to still be able to recognize labels that are
- After that. The next step is to convert tflites without metadata into metadata tflites. After you download this repository into a zip and there is a file called "TFLITE Converter". And just open the file there is some code you will change
- Then, you will change some of the code:
  * This changes the name of your tflite file that doesn't have metadata yet
  ```bash
  flags.DEFINE_string("model_file", "HayModels.tflite",
                        "Path and file name to the TFLite model file.")
  ```
  * Change this local directory, so that it can be exported directly
  ```bash
  flags.DEFINE_string("export_directory", "D:/MODELS/",
                      "Path to save the TFLite model files with metadata.")
  ```
  * Modify this specification model according to the model you created earlier. 
  ```bash
  flags.DEFINE_string("export_directory", "D:/MODELS/",
                      "Path to save the TFLite model files with metadata.")"Hay-Models.tflite":
        ModelSpecificInfo(
            name="Hay Model Image Classifier",
            version="v1",
            image_width=224,  # Replace with your model's input width
            image_height=224,  # Replace with your model's input height
            image_min=0,
            image_max=255,
            mean=[127.5],  # Replace with your model's mean if different
            std=[127.5],  # Replace with your model's std if different
            num_classes=12,  # Replace with your model's number of classes
            author=""
        )
  ```
- That's some code that needs to be changed, after that run this file and the file will automatically convert itself with the metadata.
- After that, this tflite file will be used by Mobile Development.

## Authors
* **Muhammad Haris**  - [muhdharis28](https://github.com/muhdharis28)
* **Irfan Saputra Nasution**  - [irfansaputranst](https://github.com/irfansaputranst)
* **Mutiara Citra**  - [Mutiaracitra](https://github.com/Mutiaracitra)

