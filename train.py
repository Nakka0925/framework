import numpy as np
import yaml
import pandas as pd
from tensorflow.keras import layers, models
from tensorflow.keras.utils import plot_model
from tools.dataset import dataset_gain
from tools.f1score_gen import f1score_cacl
from tools.data_generator import ImageDataGenerator
from pathlib import Path


with open('train_setting.yml', 'r') as yml:
    config = yaml.safe_load(yml)

all_loss = []
all_val_loss = []
all_accuracy = []
all_val_accuracy = []
all_f1score = []

csv_dst = Path(config["destination"]) / "cross_val" / "csv"
csv_dst.mkdir(parents=True, exist_ok=True)

#交差検証
######################################################################
for idx in range(1,config['fold_num']+1):

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
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(51, activation='softmax'))

    plot_model(model, show_shapes=True, to_file='model.png')

    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])


    train_x, train_t, test_x, test_t = dataset_gain(config['destination'], config['creature_data_destination'], config['fold_num'], idx)

    history = model.fit_generator(
        generator=train_datagen.flow_from_directory(train_x, train_t, config['batch_size']),
        steps_per_epoch=int(np.ceil(len(train_x) / config['batch_size'])),
        epochs=config['epochs'],
        verbose=1,
        validation_data=test_datagen.flow_from_directory(test_x, test_t),
        validation_steps=int(np.ceil(len(test_x) / config['batch_size']))
        )

    model.save('saved_model/my_model')

    history_dict = history.history

    acc = history_dict['accuracy']
    val_acc = history_dict['val_accuracy']
    loss = history_dict['loss']
    val_loss = history_dict['val_loss']

    #学習データをcsv化
    data = {'accuracy' : acc, 'val_accuracy' : val_acc, 'loss' : loss, 'val_loss' : val_loss}
    df = pd.DataFrame(data, columns=['accuracy', 'val_accuracy', 'loss', 'val_loss'])
    df.index = ['epoch ' + str(n) for n in range(1, config['epochs']+1)]
    df.to_csv(csv_dst / (config['accuracy_loss_dataname'] + str(idx) + '.csv'))

    all_accuracy.append(acc)
    all_val_accuracy.append(val_acc)
    all_loss.append(loss)
    all_val_loss.append(val_loss)
    all_f1score.append(f1score_cacl(test_x, test_t, model, idx))
######################################################################

#accuracy, lossの平均
ave_accuracy = np.mean(all_accuracy, axis = 0)
ave_val_accuracy = np.mean(all_val_accuracy, axis = 0)
ave_loss = np.mean(all_loss, axis = 0)
ave_val_loss = np.mean(all_val_loss, axis = 0)

data = {'accuracy' : ave_accuracy, 'val_accuracy' : ave_val_accuracy, 'loss' : ave_loss, 'val_loss' : ave_val_loss}
df = pd.DataFrame(data, columns=['accuracy', 'val_accuracy', 'loss', 'val_loss'])
df.index = ['epoch ' + str(n) for n in range(1, config['epochs']+1)]
df.to_csv(csv_dst / (config['accuracy_loss_dataname'] + "_all.csv"))

#f1score
data = {'f1score' : all_f1score}
df = pd.DataFrame(data, columns=['f1score'])
df.index = ['k' + str(n) for n in range(1, config['fold_num']+1)]
df.to_csv(csv_dst / (config['f1score_dataname'] + ".csv"))