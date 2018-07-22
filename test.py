import keras
from keras.models import load_model
# load model
model = load_model('model1_cnn.h5')

from keras.preprocessing.image import ImageDataGenerator
test_datagen = ImageDataGenerator(rescale = 1./255)
test_generator = test_datagen.flow_from_directory(
    'New Folder',
    target_size = (128, 128),  #input dim
    color_mode = 'rgb',
    batch_size = 1,
    class_mode = 'categorical',
    shuffle = False)

#test predictions with generator
test_files_names = test_generator.filenames
predictions = model.predict_generator(test_generator)

import pandas as pd
import numpy as np

dataset = pd.read_csv('labels.csv')
img_labels = list(dataset['breed'].values)
folders_to_be_created = np.unique(dataset.iloc[:,1].values)

predictions_df = pd.DataFrame(predictions, columns = list(folders_to_be_created))
predictions_df.insert(0, "id", test_files_names)
predictions_df['id'] = predictions_df['id'].map(lambda x: x.replace('test\\','').replace('.jpg', ''))
#predictions_df['id'] = predictions_df['id'].map(lambda x: x.lstrip('test\\').rstrip('.jpg'))
#why is e getting removed??? - because lstrip works as whatever the characters are in the given string they'll be stripped off from the left no matter what order hence e was getting removed!
predictions_df.to_csv('model1_predictions.csv', index = False)




