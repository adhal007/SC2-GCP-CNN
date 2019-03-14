import argparse
import model
import data_utils
import os
import random
import numpy as np
import tensorflow as tf
import keras
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
import np_utils
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import os
from keras.utils import to_categorical
# from tensorflow import tensorboard
## set first task as training data building and create training and testing from load args here
### testing test_build functions
path = "C:/Users/adhal/SC2-GCP-Training/training-data-2"
os.chdir(path)
train_data_dir = path

def check_data(no_attacks, attack_closest_to_nexus, attack_enemy_structures, attack_enemy_start):
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

def build_train_data(train_data_dir):
    current = 0
    increment = 25
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
                print(d[0])
                choice = np.argmax(d[0])
                if choice == 0:
                    no_attacks.append([d[0], d[1]])
                elif choice == 1:
                    attack_closest_to_nexus.append([d[0], d[1]])
                elif choice == 2:
                    attack_enemy_structures.append([d[0], d[1]])
                elif choice == 3:
                    attack_enemy_start.append([d[0], d[1]])
            current += increment
            if current > maximum:
                not_maximum = False

        lengths = check_data(no_attacks, attack_closest_to_nexus, attack_enemy_structures, attack_enemy_start)
        lowest_data = min(lengths)

        random.shuffle(no_attacks)
        random.shuffle(attack_closest_to_nexus)
        random.shuffle(attack_enemy_structures)
        random.shuffle(attack_enemy_start)

        no_attacks = no_attacks[:lowest_data]
        attack_closest_to_nexus = attack_closest_to_nexus[:lowest_data]
        attack_enemy_structures = attack_enemy_structures[:lowest_data]
        attack_enemy_start = attack_enemy_start[:lowest_data]

        lengths_total = check_data(no_attacks, attack_closest_to_nexus, attack_enemy_structures, attack_enemy_start)
        train_data = no_attacks + attack_closest_to_nexus + attack_enemy_structures + attack_enemy_start
        train_data_np = np.array(train_data)
        print(np.shape(train_data_np[0]))
        random.shuffle(train_data)
        print(len(train_data))
        training_data_lengths = len(train_data)
        np.save('../all_raw_data/SC2-all-data-2.npy', train_data)
        return train_data, lengths_total
# set batch size and epochs here
def train_model(args):
    ## test print size of input data
    train_data, lengths_total = build_train_data("C:/Users/adhal/SC2-GCP-Training/training-data-2/")
    total = 0
    for i in range(len(lengths_total)):
        total += lengths_total[i]
    print(total)
    SC2_model = model.SC2_model()
    # ## load built dataset
    train_features, test_features, train_labels, test_labels = \
        data_utils.load_data(args)


# convert from int to float
    train_features = train_features.astype('float32')
    test_features = test_features.astype('float32')

    SC2_model.train_on_batch(train_features, train_labels)
    scores = model.evaluate(test_f, test_f, verbose=1)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])



def get_args():
    parser = argparse.ArgumentParser(description='Starcraft2 keras example')
    parser.add_argument('--model-dir',
                        type=str,
                        help='Where to save the model')
    parser.add_argument('--model-name',
                        type=str,
                        default='SC2_model.h5',
                        help='What to name the saved model file')
    parser.add_argument('--batch-size',
                        type=int,
                        default=4,
                        help='input batch size for training (default: 4)')
    parser.add_argument('--test-split',
                        type=float,
                        default=0.2,
                        help='split size for training / testing dataset')
    parser.add_argument('--epochs',
                        type=int,
                        default=10,
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help='random seed (default: 42)')
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    train_model(args)


if __name__ == '__main__':
    main()
