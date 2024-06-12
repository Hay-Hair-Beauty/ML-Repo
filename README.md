# Hay Healthy - Scan Hair Predict
This is a Capstone Project part Machine Learning-- <br/>
We created a Tensorflow model to predict hair on a person. Thus, we hope to easily predict whether a user has a hair disease or not.

## Project Structure
```bash
.
├── README.md
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
├── .gitattributes "--> sending large files from tflite via Github Large File"
├── Hay-Models.h5 "--> a convert model to TFDH5 file for mobile"
├── Hay-Models.tflite "--> a convert model from TFDH5 to tflite file for mobile"
├── HayModels-Baseline.ipynb "--> a baseline notebook python"
├── data_recomm.csv "--> a data set that recommends a shampoo/treatment"
├── label.txt "--> label of a hair disease"
└── recommender_system.py "--> a file of recommender system"
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
- Copy this repository link into the collab https://github.com/Hay-Hair-Beauty/ML-Repo.git 
- After succesfully open the notebook, download dataset from kaggle

## Built With
* [Tensorflow Keras](https://www.tensrflow.org) - The AI framework used

## Authors
* **Muhammad Haris**  - [muhdharis28](https://github.com/muhdharis28)
* **Irfan Saputra Nasution**  - [irfansaputranst](https://github.com/irfansaputranst)
* **Mutiara Citra**  - [Mutiaracitra](https://github.com/Mutiaracitra)
