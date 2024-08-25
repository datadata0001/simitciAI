import tensorflow as tf

if tf.config.list_physical_devices('GPU'):
    print("TensorFlow GPU'yu kullanıyor.")
else:
    print("TensorFlow GPU'yu kullanmıyor.")

from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras.models import Sequential
from tensorflow.keras.metrics import Precision, Recall
import numpy as np
# ilkleme
classifier = Sequential()

# Adım 1 - Convolution
classifier.add(Convolution2D(32, 3, 3, input_shape = (128, 128, 3), activation = 'relu'))

# Adım 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# 2. convolution katmanı
classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adım 3 - Flattening
classifier.add(Flatten())

# Adım 4 - YSA
classifier.add(Dense(units = 512, activation = 'linear'))

classifier.add(Dense(units = 512, activation = 'linear'))

classifier.add(Dense(units = 512, activation = 'linear'))

classifier.add(Dense(units = 512, activation = 'linear'))

classifier.add(Dense(units = 3, activation = 'softmax'))

# CNN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy',Precision(name='precision'),Recall(name='recall')])

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

# Sınıf ağırlıklarını hesapla
from sklearn.utils import class_weight
class_weights = class_weight.compute_class_weight('balanced',
                                                  classes=np.unique(training_set.classes),
                                                  y=training_set.classes)
class_weights = dict(enumerate(class_weights))


classifier.fit(training_set,
                         #steps_per_epoch = 8000,
                         epochs = 100,
                         validation_data = test_set,
                         #validation_steps = 2000),
                         )


# Test setini değerlendirme
test_loss, test_accuracy, test_precision, test_recall = classifier.evaluate(test_set)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")
print(f"Test Precision: {test_precision}")
print(f"Test Recall: {test_recall}")


classifier.summary()

# ValGen'den bir batch alın
val_images, val_labels = next(test_set)

# Tahmin yapın
predictions = classifier.predict(val_images)

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



#Gerçek değerler vs Tahmini değerler

import scipy.stats as stats
import matplotlib.pyplot as plt

plt.figure(figsize =(12,6))

#Gerçek Değerler
plt.subplot(1, 2, 1)
stats.probplot(true_classes , dist = "norm", plot = plt)
plt.title("Gerçek Değerler")


#Tahmini Değerler
plt.subplot(1, 2, 2)
stats.probplot(pred_classes , dist = "norm", plot = plt)
plt.title("Tahmini Değerler")

plt.tight_layout()
plt.show()


#Modeli Kaydetme 
classifier.save('model_deneme.h5')


#################################################################
#Kayitli Modeli Test Etme
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




#OpenCV fotograf ile test etme

import cv2
from tensorflow.keras.preprocessing.image import img_to_array
import os 
import random

folder_path = 'test'
class_labels = list(training_set.class_indices.keys())
# Klasördeki tüm dosyaların listesini al
file_group_list = os.listdir(folder_path) #Hangi teror orgutune mensup

random_file_group = random.choice(file_group_list) #Rastgele teror orgutu sec
print(f'Real Conclusion: {random_file_group}')

random_file_group_path = os.path.join(folder_path, random_file_group) #Secilen teror orgutunun yolu

file_list = os.listdir(random_file_group_path) #Fotograflarin listesi

random_file = random.choice(file_list)#Rastgele fotograf sec

random_file_path = os.path.join(random_file_group_path, random_file)#Tam dosya yolu

cap = cv2.imread(random_file_path)
resized_frame = cv2.resize(cap,(128,128))

img_array = img_to_array(resized_frame) / 255.0
img_array = np.expand_dims(img_array, axis = 0) #batch icin ekstra boyut ekleme

predictions_img = loaded_model.predict(img_array)
print(f'Oranlar{predictions_img}')

class_idx = np.argmax(predictions_img[0])
class_name = class_labels[class_idx]
print(f'Predict Conclusion: {class_name}')

img = cv2.putText(cap, f"Predictions: {class_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

cv2.imshow('Test Image', cap)


cv2.waitKey(0)

cv2.destroyAllWindows()



"""
###########################################
#OpenCV
#Ekrani Goruntusu
import cv2
import numpy as np
import pyautogui
from tensorflow.keras.preprocessing.image import img_to_array

region = (0, 0, 1440, 960) #Ekran genisligi icin verilmis kod.

font = cv2.FONT_HERSHEY_SIMPLEX

class_labels = list(training_set.class_indices.keys())

while True:
    
    #Erkan Goruntusu
    img = pyautogui.screenshot(region=region)

    #OpenCV formatina donusturmek
    frame = np.array(img)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    #Kareyi 64x64 ayarliyoruz
    resized_frame = cv2.resize(frame,(128,128))
    
    #Goruntuyu model girisi icin hazirlamak
    img_array = img_to_array(resized_frame) / 255.0
    img_array = np.expand_dims(img_array, axis = 0)
    
    #Model ile tahmin
    predictions_cv2 = loaded_model.predict(img_array)
    class_idx = np.argmax(predictions_cv2[0])
    print(f'predictions {predictions_cv2}')
    print(f'class pred {class_idx}')

    
    #Sinif isimlerini al
    class_name = class_labels[class_idx]
    
    #Tahmini ekranda goster
    cv2.putText(frame,f'Class:{class_name}',(10,30),font,1,(0,255,0),2,cv2.LINE_AA)
    
    #Ekrani goster
    cv2.imshow('Screen',frame)
    
    #q tusuna basinca biter
    if cv2.waitKey(1) & 0xff == ord('q'):
        break


#Pencereleri ucurur!!
cv2.destroyAllWindows()



"""


