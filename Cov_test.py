import numpy as np
#import tensorflow as tf
from keras import layers, models
#from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
#from tensorflow.keras.utils import plot_model
from dataset import dataset_gain
import yaml
from data_generator import ImageDataGenerator


train_datagen = ImageDataGenerator()
test_datagen = ImageDataGenerator()

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(192, 192, 1)))
model.add(layers.Dropout(0.3))
"""
model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.Dropout(0.3))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Dropout(0.3))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Dropout(0.3))
"""
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())

model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(49, activation='softmax'))

#plot_model(model, show_shapes=True, to_file='model.png')

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

with open('train_setting.yml', 'r') as yml:
    config = yaml.safe_load(yml)

#交差検証
######################################################################
#for idx in range(1,config['fold_num']+1):

train_x, train_t, test_x, test_t = dataset_gain(config['destination'], config['creature_data_destination'], config['fold_num'], 1)

history = model.fit_generator(
    generator=train_datagen.flow_from_directory(train_x,train_t),
    steps_per_epoch=int(np.ceil(len(train_x) / config['batch_size'])),
    epochs=config['epochs'],
    verbose=1,
    validation_data=test_datagen.flow_from_directory(test_x, test_t),
    validation_steps=int(np.ceil(len(test_x) / config['batch_size']))
    )




model.save('saved_model/my_model')

#test_loss, test_acc = model.evaluate(test_x,  test_t, verbose=1)

#print('\nTest accuracy:', test_acc)
#print('\nTest loss:', test_loss)
#predictions = model.predict(test_x)

#print(predictions[0])

history_dict = history.history
#print (history_dict.keys())

import matplotlib.pyplot as plt

acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)


#plt.plot(epochs, loss, marker="o", label='Training loss')
plt.plot(epochs, val_loss, marker="o")
plt.title('loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
#plt.ylim(0.0, 1.5)
plt.legend()
plt.savefig("loss_low.png")
#plt.show()

plt.clf()  # 図のクリア

#plt.plot(epochs, acc, marker="o", label='Training acc')
plt.plot(epochs, val_acc, marker="o", color="orangered")
plt.title('accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.ylim(0.0, 1.0)
plt.legend()
plt.savefig("accuracy_low.png")
#plt.show()