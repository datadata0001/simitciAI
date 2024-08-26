# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 14:38:57 2024

@author: orhan
"""


from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras.models import Sequential
from tensorflow.keras.metrics import Precision, Recall
import numpy as np


# CNN ve resimler

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True,
                                   rotation_range = 20,
                                   width_shift_range = 0.2,
                                   height_shift_range = 0.2,
                                   brightness_range =[0.5, 1.5],
                                   validation_split= 0.2)



training_set = train_datagen.flow_from_directory('pkkfeto',
                                                 target_size = (128, 128),
                                                batch_size = 1024,
                                                 class_mode = 'categorical',
                                                 subset='training')

test_set = train_datagen.flow_from_directory('pkkfeto',
                                            target_size = (128, 128),
                                                batch_size = 1024,
                                            class_mode = 'categorical',
                                            subset='validation')


from keras.models import load_model
loaded_model = load_model('model_deneme.h5')



# ValGen'den bir batch alın
val_images, val_labels = next(test_set)

# Tahmin yapın
predictions = loaded_model.predict(val_images)

# Tahminlerin ve gerçek etiketlerin ilk birkaçını yazdırın
pred_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(val_labels, axis=1)

print("Gerçek Etiketler: ", true_classes)
print("Tahminler: ", pred_classes)

# Modeli değerlendirme

from sklearn.metrics import accuracy_score, f1_score, r2_score

accuracy = accuracy_score(true_classes, pred_classes)
print(f"Doğruluk (Accuracy): {accuracy}")

f1 = f1_score(true_classes, pred_classes, average='weighted') #İkili Sınıflandırma problemlerini test etmek için daha uygun bir yapı.
print(f"F1 Skoru: {f1}")



#OpenCV secilen fotograf ile test etme

import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
import os 
import random

folder_path = 'dosya' #----------------------------------------> Tahmin ettirmek istedigniz dosya adini giriniz.
class_labels = list(training_set.class_indices.keys())
# Klasördeki tüm dosyaların listesini al
file_group_list = os.listdir(folder_path) #Hangi teror orgutune mensup

take_cap = 'terörist1.jpg' #----------------------------------------> Istediginiz resmin dosyasinizda ki ismini burada belirtiniz.

file_path = os.path.join(folder_path, take_cap)#Tam dosya yolu

cap = cv2.imread(file_path)

resized_frame = cv2.resize(cap,(128,128))

img_array = img_to_array(resized_frame) / 255.0
img_array = np.expand_dims(img_array, axis = 0) #batch icin ekstra boyut ekleme

predictions_img = loaded_model.predict(img_array)
print(f'Oranlar{predictions_img}')
class_idx = np.argmax(predictions_img[0])
class_name = class_labels[class_idx]
print(f'Predict Conclusion: {class_name}')

img = cv2.putText(cap, f"Predictions: {class_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

cv2.imshow('Test Image', img)


cv2.waitKey(0)

cv2.destroyAllWindows()
