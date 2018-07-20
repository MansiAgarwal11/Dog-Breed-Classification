#data augmentation to prevent overfitting and generaton of more data for better results
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

training_set = train_datagen.flow_from_directory('train',
                                                 target_size = (128, 128),
                                                 color_mode= 'rgb',
                                                 batch_size = 16)  

#training_set.class_indices

#Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Convolution2D(64, 3, 3, input_shape = (128, 128, 3), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Convolution2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Dropout(0.25))

#Adding a third convolutional layer
classifier.add(Convolution2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(output_dim = 200, activation = 'relu'))
classifier.add(Dense(120, activation = 'softmax'))

# Compiling the CNN
from keras import optimizers
adam = optimizers.Adam()
classifier.compile(adam, loss = 'categorical_crossentropy', metrics = ['accuracy'])

print(classifier.summary())
classifier.fit_generator(training_set,
                         samples_per_epoch = 10222,
                         nb_epoch = 25)

#Saving the model
#saves architecture, weights, state, configuration of the model
classifier.save('model1_cnn.h5')  


#import h5py
#model_json = classifier.to_json()
#with open("model1_cnn.json", "w") as json_file:
#    json_file.write(model_json)
#classifier.save_weights('model1_cnn.h5/weights') 



