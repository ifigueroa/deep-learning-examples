# Convolutional Neural Network

"""
 The data pre-processing is just splitting the data into
 train, test and perhaps single predictions
""" 

# Part 1 - Building the CNN

# Importing the KEras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
"""
 32 3x3 feature detectors
 input_shape = All the images need to be the same size
   The order in the TensorFlow backend is different than in Theano
   In TensoFlow is (width, length, channels)
 The activation layer is ReLU to avoid negative pixel values
"""
classifier.add(Convolution2D(32, 3, 3,
                             input_shape = (64, 64, 3),
                             activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
"""
 Rule of thumbs for the number of nodes:
 Somewhere between the num of inputs and the number of outputs
 This number is a result of experimentation
"""
classifier.add(Dense(output_dim = 128,
                     activation = 'relu'))
"""
 Now add the sigmoid function (binary output)
 for more classes, we should use softmax
"""
classifier.add(Dense(output_dim = 1,
                     activation = 'sigmoid'))

# Compile the CNN
classifier.compile(optimizer = 'adam',
                   loss = 'binary_crossentropy',
                   metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images
"""
  We need to do image augmentation to prevent over-fitting.
  When there's not a lot of images, the model does not generalize
  so well. 
"""
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

classifier.fit_generator(training_set,
                         samples_per_epoch = 8000,
                         nb_epoch = 25,
                         validation_data = test_set,
                         nb_val_samples = 2000
                         )































