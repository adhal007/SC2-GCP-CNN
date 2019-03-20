import keras  # Keras 2.1.2 and TF-GPU 1.9.0
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import TensorBoard
import numpy as np
import os
import random
from keras.callbacks import CSVLogger
import keras.backend as K

def mean_pred(y_true, y_pred):
    return K.mean(y_pred)

def recall(y_true, y_pred):
    #Recall metric.

    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)),axis=0)
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)),axis=0)
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision(y_true, y_pred):
    #Precision metric.

    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)),axis=0)
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)),axis=1)
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

model = Sequential()

model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=(176, 200, 3),
                 activation='relu'))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(64, (3, 3), padding='same',
                 activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(128, (3, 3), padding='same',
                 activation='relu'))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4, activation='softmax'))

learning_rate = 0.0001
opt = keras.optimizers.adam(lr=learning_rate, decay=1e-6)

model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy', mean_pred, recall, precision])

# tensorboard = TensorBoard(log_dir="/content/drive/My Drive/logs/")
train_data_dir = "/content/drive/My Drive/train_data"
with open('report.txt','w') as fh:
    # Pass the file handle in as a lambda function to make it callable
    model.summary(print_fn=lambda x: fh.write(x + '\n'))

def check_data():
    choices = {"no_attacks": no_attacks,
               "attack_closest_to_nexus": attack_closest_to_nexus,
               "attack_enemy_structures": attack_enemy_structures,
               "attack_enemy_start": attack_enemy_start}

    total_data = 0

    lengths = []
    for choice in choices:
        print("Length of {} is: {}".format(choice, len(choices[choice])))
        total_data += len(choices[choice])
        lengths.append(len(choices[choice]))

    print("Total data length now is:", total_data)
    return lengths


hm_epochs = 100

for i in range(hm_epochs):
    current = 0
    increment = 75
    not_maximum = True
    all_files = os.listdir(train_data_dir)
    maximum = len(all_files)
    random.shuffle(all_files)

    while not_maximum:
        print("WORKING ON {}:{}".format(current, current+increment))
        no_attacks = []
        attack_closest_to_nexus = []
        attack_enemy_structures = []
        attack_enemy_start = []

        for file in all_files[current:current+increment]:
            full_path = os.path.join(train_data_dir, file)
            data = np.load(full_path)
            data = list(data)
            for d in data:
                choice = np.argmax(d[0])
                if choice == 0:
                    no_attacks.append([d[0], d[1]])
                elif choice == 1:
                    attack_closest_to_nexus.append([d[0], d[1]])
                elif choice == 2:
                    attack_enemy_structures.append([d[0], d[1]])
                elif choice == 3:
                    attack_enemy_start.append([d[0], d[1]])

        lengths = check_data()
        lowest_data = min(lengths)

        random.shuffle(no_attacks)
        random.shuffle(attack_closest_to_nexus)
        random.shuffle(attack_enemy_structures)
        random.shuffle(attack_enemy_start)

        no_attacks = no_attacks[:lowest_data]
        attack_closest_to_nexus = attack_closest_to_nexus[:lowest_data]
        attack_enemy_structures = attack_enemy_structures[:lowest_data]
        attack_enemy_start = attack_enemy_start[:lowest_data]

        check_data()

        train_data = no_attacks + attack_closest_to_nexus + attack_enemy_structures + attack_enemy_start

        random.shuffle(train_data)
        print(len(train_data))

        test_size = 300
        batch_size = 4

        x_train = np.array([i[1] for i in train_data[:-test_size]]).reshape(-1, 176, 200, 3)
        y_train = np.array([i[0] for i in train_data[:-test_size]])

        x_test = np.array([i[1] for i in train_data[-test_size:]]).reshape(-1, 176, 200, 3)
        y_test = np.array([i[0] for i in train_data[-test_size:]])

        csv_logger = CSVLogger('/content/drive/My Drive/log200.csv', append=True, separator=';')
        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  validation_data=(x_test, y_test),
                  shuffle=True,
                  verbose=1, callbacks=[csv_logger])

        # fit


        current = current + increment
        if current > 150:
              not_maximum = False
