import numpy as np
#import tensorflow. as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from tensorflow.keras.utils import plot_model
from dataset import dataset_gain
import yaml, cv2
from data_generator import ImageDataGenerator
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sns
from pathlib import Path

train_datagen = ImageDataGenerator()
test_datagen = ImageDataGenerator()

model = models.Sequential()
model.add(layers.Conv2D(8, (3, 3), activation='relu', padding='same', input_shape=(192, 192, 1)))
model.add(layers.Conv2D(8, (3, 3), padding='same', activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(16, (3, 3), padding='same', activation='relu'))
model.add(layers.Conv2D(16, (3, 3), padding='same', activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(49, activation='softmax'))

plot_model(model, show_shapes=True, to_file='model.png')

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
    generator=train_datagen.flow_from_directory(train_x, train_t, config['batch_size']),
    steps_per_epoch=int(np.ceil(len(train_x) / config['batch_size'])),
    epochs=config['epochs'],
    verbose=1,
    validation_data=test_datagen.flow_from_directory(test_x, test_t),
    validation_steps=int(np.ceil(len(test_x) / config['batch_size']))
    )


model.save('saved_model/my_model')

images = []

for path in test_x:
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    images.append(1 - np.asarray(img, dtype=np.float32) / 255)

images = np.asarray(images, dtype=np.float32)
test_t = np.asarray(test_t, dtype=np.float32)

test_loss, test_acc = model.evaluate(images,  test_t, verbose=1)

predictions = model.predict(images)
classes_x = np.argmax(predictions,axis=1)

tmp = confusion_matrix(test_t, classes_x)

tmp2 = np.sum(tmp, axis=1)
tmp = [tmp[idx] / tmp2[idx] for idx in range(49)]

tmp = np.round(tmp, decimals=2)
    
with open('train_setting.yml', 'r') as yml:
    config = yaml.safe_load(yml)

graph_dst = Path(config['destination']) / config['data_division'] / 'graph'
graph_dst.mkdir(parents=True, exist_ok=True)
graph_dst = graph_dst / 'confusion_matrix'
graph_dst.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(config['creature_data_destination'] + 'csv/class_sum.csv', encoding='shift-jis')
fig, axes = plt.subplots(figsize=(22,23))
cm = pd.DataFrame(data=tmp, index=df['class'], columns=df['class'])

sns.heatmap(cm, square=True, cbar=True, annot=True, cmap = 'Blues', vmax=1, vmin=0)
plt.xlabel("Pre", fontsize=15, rotation=0)
plt.xticks(fontsize=15)
plt.ylabel("Ans", fontsize=15)
plt.yticks(fontsize=15)
plt.savefig(graph_dst / (config['heat_map_name'] + str(1) + '.png')) 
plt.close()






history_dict = history.history

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