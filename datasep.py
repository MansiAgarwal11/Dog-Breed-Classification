import pandas as pd
import os
import numpy as np
import shutil

# source is the current directory
# Open dataset file
dataset = pd.read_csv('labels.csv')
file_names = list(dataset['id'].values)
img_labels = list(dataset['breed'].values)


folders_to_be_created = np.unique(dataset.iloc[:,1].values)

source = os.getcwd()

for new_path in folders_to_be_created:
    if not os.path.exists(source + '/' + new_path):
        os.makedirs(new_path)


folders = folders_to_be_created.copy()

for f in range(len(file_names)):  

  current_img = file_names[f]
  current_label = img_labels[f]
  
  s = source + '\\new_train\\' + current_img + '.jpg' #"C:\Users\Mansi Agarwal\Desktop\dog breed classification\new_train\000bec180eb18c7604dcecc8fe0dba07.jpg"
  d = source + '\\' + current_label
  shutil.move(s,d)

