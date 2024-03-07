
# ProjectTitle: MULTICLASS-DISEASE-CLASSIFICATION-OF-MEDICAL-IMAGE-DATA-USING-CONVOLUTIONAL-NEURAL-NETWORK
Dataset links:

* GastroIntestinal Disease(8 Classes) : https://datasets.simula.no/kvasir/
* Potato(3 Classes) & Corn Plant(4 Classes) leaf Disease : https://www.kaggle.com/datasets/abdallahalidevplantvillage-dataset
* Pneumonia(Binary Class) :https://www.kaggle.com/datasets/tolgadincer/labeled-chest-xray-images


## OBJECTIVE:

  The objectives of the BCG  Virtual Experience Program:

      1. The primary objective of this project was to develop robust classification models capable of accurately identifying various diseases from medical images, utilizing convolutional neural networks (CNNs) for Binary Class(Pneumonia/Normal), 3 Classes( Potato Plant Leaf Disease), 4 Classes(Corn Plant Leaf Disease) and 8 Class(GastroIntestinal Diseases in Humans).
      2. To learn about Transfer Learning for Deep learning models to classify the complex image dataset. Evaluate and fine-tune models for accuracy and effectiveness based on its classes.
      3. Interpret model results to identify key factors.

## Classes with No of images:

**BINARY CLASS IMAGE CLASSIFICATION-PNEUMONIA X-RAY CNN MODELLING**

* PNEUMONIA - 4273 Images
* Normal - 1583 Images

**3 CLASS IMAGE CLASSIFICATION-POTATO PLANT LEAF DISEASES CNN MODELLING**

* Potato___Early_blight - 1000 Images
* Potato___Late_blight - 1000 Images
* Potato___healthy - 1000 Images

**4 CLASS IMAGE CLASSIFICATION-CORN PLANT LEAF CNN MODELLING**

* Corn___Cercospora_leaf_spot Gray_leaf_spot	- 1000 Images
* Corn___Common_rust	- 1192 Images
* Corn___Northern_Leaf_Blight - 1000 Images
* Corn___healthy - 1162 Images

**8 CLASS IMAGE CLASSIFICATION-GASTRO INTESTINAL DISEASES CNN MODELLING WITH TRANSFER LEARNING VGG19**

* dyed-lifted-polyps	- 1000 Images
* dyed-resection-margins	- 1000 Images
* esophagitis	- 1000 Images
* normal-cecum	- 1000 Images
* normal-pylorus	- 1000 Images
* normal-z-line	- 1000 Images
* polyps	- 1000 Images
* ulcerative-colitis	- 1000 Images

## LIBRARIES USED:

   - Pandas, Matplotlib, Seaborn, Scikit-learn, Cv2, Pathlib, Tensorflow, Numpy, Keras.


## TABLE OF CONTENTS:

- [Introduction](#introduction)
- [Data Preprocessing](#data-preprocessing)
- [Model Building](#Model-Building)
- [Model Evaluation Metircs](#Model-Evaluation-Metircs)
- [Insights](#insights)
      
## Introduction:

  Medical imaging plays a pivotal role in diagnosing and understanding various diseases across different domains, ranging from human health to agriculture. With the advancement of deep learning techniques, particularly Convolutional Neural Networks (CNNs), the ability to analyze and interpret medical images has been revolutionized, enabling more accurate and efficient disease identification and classification. In this project, we embark on a comprehensive journey to develop robust classification models leveraging CNNs for analyzing medical image data across multiple domains. Our primary objective is to accurately identify and classify different diseases represented in the datasets provided. These datasets encompass a diverse range of medical image data, including gastrointestinal diseases in humans, potato plant leaf diseases, corn plant leaf diseases, and pneumonia diagnosis from chest X-rays.

## Data Preprocessing:

   Image is an unstructured data where it requires additional caution while preprocesing part. First, I tried to load the image folders directory to the analysis area. Using the pathlib library of python tried to know about the images distributions across the different folders and tried to know about no of images in each folders of the classes. Then, Using the Cv2 library helps to read the images in the class folders. Before the modelling part starts images should be resized(WIDTH X HEIGHT) in a proper dimensions to extract the pixel values for analysis.

   * Pneumonia(Binary Class) : (200 X 200)
   * Potato Plant(3 Classes) leaf Disease : (100 x 100)
   * & Corn Plant(4 Classes) leaf Disease :  (100 x 100)
   * GastroIntestinal Disease(8 Classes) : (100 X 100)

After resizing the images to required dimensions splitted the Images pixel values  as X and Image Categoris/classes as Y for modelling using Numpy. For X we need to rehape it again due to the Image is actually in 3 Color Channels called Red, Green , Blue. So, it should in proper format before Cnn model building.

   * Pneumonia(Binary Class) : (200 X 200 x 3)
   * Potato Plant(3 Classes) leaf Disease : (100 x 100 x 3)
   * & Corn Plant(4 Classes) leaf Disease :  (100 x 100  x 3)
   * GastroIntestinal Disease(8 Classes) : (100 X 100 x 3) for each Image pixels values stored as X. Where Y will represents the Category/Class label of the respective Images.

Then Visualized the Images from different classes with its labels using Matplotlib and cv2 library.
Now, time to split the dataset into Train/Val/Test split in the ratio of  70% / 20% / 10%.  Using the tensorflow function called 'to_categorical' helped for converting to one-hot encoding categorical Classes for Y.

Images Pixels values should be preproceesed in way called Normalizing the pixels values by dividing 255, Some random horizontal and vertical flips , random level of zoomings of the images pixels must be handled before modelling. These things will be taken care of using the ImageDataGenerators which will handle effectively. At, last fit the genrators for the Train-Val-Test sets.

## Model Building

   For CNN modell building part, some basics needed to be known prior about how convolutions works.

   * Convolutional Layer: The convolutional layer is the first layer in a CNN that extracts various features from input images. It achieves this by performing mathematical operations called convolutions between the input image and a filter of a specific size (MxM).

   * Convolution Operation: The convolution operation involves sliding the filter over the input image and computing the dot product between the filter and parts of the input image corresponding to its size. This process is applied across the entire image.

   * Feature Map: The output of the convolutional layer is termed as the feature map. It provides information about the image, such as edges, corners, and other patterns. Each feature map corresponds to a specific filter applied to the input image.

   * Pooling Layer: A Convolutional Layer is followed by a Pooling Layer. The primary aim of this layer is to decrease the size of the convolved feature map to reduce the computational costs. This is performed by decreasing the connections between layers and independently operates on each feature map. Depending upon method used, there are several types of Pooling operations. It basically summarises the features generated by a convolution layer.

   * Fully Connected Layer: The Fully Connected (FC) layer consists of the weights and biases along with the neurons and is used to connect the neurons between two different layers. These layers are usually placed before the output layer and form the last few layers of a CNN Architecture.

   * Dropout:  when all the features are connected to the FC layer, it can cause overfitting in the training dataset. Overfitting occurs when a particular model works so well on the training data causing a negative impact in the model’s performance when used on a new data. Dropout results in improving the performance of a machine learning model as it prevents overfitting by making the network simpler. It drops neurons from the neural networks during training.

   * Activation Functions: It decides which information of the model should fire in the forward direction and which ones should not at the end of the network.It adds non-linearity to the network. There are several commonly used activation functions such as the ReLU, Softmax, tanH and the Sigmoid functions. Each of these functions have a specific usage.

   # **BINARY CLASS IMAGE CLASSIFICATION-PNEUMONIA X-RAY CNN MODELLING**

   ![screenshot](https://github.com/itsmesethus/MULTICLASS-DISEASE-CLASSIFICATION-OF-MEDICAL-IMAGE-DATA-USING-CONVOLUTIONAL-NEURAL-NETWORK-/blob/main/BINARY%20CLASS%20IMAGE%20CLASSIFICATION-PNEUMONIA%20X-RAY%20CNN%20MODELLING/img%20files/binaryclass%20model.png)

   * Activation Functions Hidden Layers: ReLU
   * Activation Function Final Layer: Sigmoid
   * Loss Function: Binary Crossentropy

   # **3 CLASS IMAGE CLASSIFICATION-POTATO PLANT LEAF DISEASES CNN MODELLING**

   ![screenshot](https://github.com/itsmesethus/MULTICLASS-DISEASE-CLASSIFICATION-OF-MEDICAL-IMAGE-DATA-USING-CONVOLUTIONAL-NEURAL-NETWORK-/blob/main/3%20CLASS%20IMAGE%20CLASSIFICATION-POTATO%20PLANT%20LEAF%20DISEASES%20CNN%20MODELLING/img%20files/3class%20model.png)

   * Activation Functions Hidden Layers: ReLU
   * Activation Function Final Layer: Softmax
   * Loss Function: Softmax Crossentropy


   # **4 CLASS IMAGE CLASSIFICATION-CORN PLANT LEAF CNN MODELLING**

   ![screenshot](https://github.com/itsmesethus/MULTICLASS-DISEASE-CLASSIFICATION-OF-MEDICAL-IMAGE-DATA-USING-CONVOLUTIONAL-NEURAL-NETWORK-/blob/main/4%20CLASS%20IMAGE%20CLASSIFICATION-CORN%20PLANT%20LEAF%20CNN%20MODELLING/img%20files/4class%20model.png)

   * Activation Functions Hidden Layers: ReLU
   * Activation Function Final Layer: Softmax
   * Loss Function: Softmax Crossentropy

   # **8 CLASS IMAGE CLASSIFICATION-GASTRO INTESTINAL DISEASES CNN MODELLING WITH TRANSFER LEARNING VGG19**

   **VGG19 Model**

   ![screenshot](https://github.com/itsmesethus/MULTICLASS-DISEASE-CLASSIFICATION-OF-MEDICAL-IMAGE-DATA-USING-CONVOLUTIONAL-NEURAL-NETWORK-/blob/main/8%20CLASS%20IMAGE%20CLASSIFICATION-GASTRO%20INTESTINAL%20DISEASES%20CNN%20MODELLING%20WITH%20TRANSFER%20LEARNING%20VGG19/img%20files/vgg19%20(4).png)


   **Final Model with VGG19 on Top**

   ![screenshot](https://github.com/itsmesethus/MULTICLASS-DISEASE-CLASSIFICATION-OF-MEDICAL-IMAGE-DATA-USING-CONVOLUTIONAL-NEURAL-NETWORK-/blob/main/8%20CLASS%20IMAGE%20CLASSIFICATION-GASTRO%20INTESTINAL%20DISEASES%20CNN%20MODELLING%20WITH%20TRANSFER%20LEARNING%20VGG19/img%20files/_final%20model%20vgg19%20(1).png)

   * Activation Functions Hidden Layers: ReLU
   * Activation Function Final Layer: Softmax
   * Loss Function: Softmax Crossentropy

## Model Evaluation Metircs

   * F1 Score
   * Precision
   * Recall
   * Accuracy

## Insights

The various CNN Architectures were built according to the different datasets used
to classify the various Classes of images based on its features. Convolutional Neural
Networks (CNNs) have proven to be a powerful tool for image classification tasks. By
using convolution layers to extract relevant features from images and pooling layers to
reduce their dimensionality, CNNs are able to learn complex representations that can
accurately classify images. And the Accuracy of the validation and test datasets are,

• Binary Image Classification (PneumoniaX-Ray Image Dataset)
 Test Accuracy = 94.88%
 Validation Accuracy = 94.31%
• Multiclass Image Classification
-Three Class Classification (Potato Plant Leaf Disease Dataset )
 Test Accuracy = 95.67%
 Validation Accuracy = 95.93%
-Four Class Classification ( Corn Plant Leaf Disease Dataset )
 Test Accuracy = 95.18%
 Validation Accuracy = 94.22%

-Eight Class Classification (Gastrointestinal Disease Dataset )
 Test Accuracy = 90.87 %
 Validation Accuracy = 89.5 %

CNNs have been successfully applied and it can be used for a wide range of
applications, including object recognition, facial recognition, and medical image analysis.
They have also achieved state-of-the-art results in several benchmark datasets.


