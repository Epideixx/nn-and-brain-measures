

# --------------------------------------------------
#               Dependencies
# --------------------------------------------------

import pandas as pd
import numpy as np
import os

import tensorflow as tf

import wandb
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint

from model import MyModel, train_model
from data import load_data, create_dataset, train_test_split
from process_data_meg import load_data_meg, get_every_phonemes_and_words, collect_activations, regression

# --------------------------------------------------

NB_LAYERS = 2
EMBEDDING_DIM_PHONES = 16
EMBEDDING_DIM_WORDS = 16
RNN_UNITS = 32
LR = 0.01
EPOCHS = 1
BATCH_SIZE = 128
TEST_SIZE = 0.2

SAVE_MODEL_PATH = 'saved_models/model'
SAVE_HIDDEN_MEMORY_FOLDER = 'saved_activations'
SAVE_PREDICTED_ENERGY_FOLDER = 'predicted_energy'

TRAINING = True
GET_ACTIVATIONS = True
PREDICT_ENERGY = True

# --------------------------------------------------

wandb.init(
    # set the wandb project where this run will be logged
    project="Text_Phoneme_Prediction_LSTM",

    # track hyperparameters and run metadata with wandb.config
    config={
        "nb_layer": NB_LAYERS,
        "rnn_units": RNN_UNITS,
        "activation_final": "softmax",
        "optimizer": "adam",
        "loss": "sparse_categorical_crossentropy",
        "learning_rate": LR,
        "epoch": EPOCHS,
        "batch_size": BATCH_SIZE,
        "embedding_dim_phones": EMBEDDING_DIM_PHONES,
        "embedding_dim_words" : EMBEDDING_DIM_WORDS,
        "test_size": TEST_SIZE,
    }
)

config = wandb.config


# --------------------------------------------------
#                   Data
# --------------------------------------------------

# Import the data 

print("Load the data...")

path_train = './data/language_model/data_combined_train.npz'
path_dev = './data/language_model/data_combined_dev.npz'
path_test = './data/language_model/data_combined_test.npz'

datafile_train = load_data(path_train)
datafile_dev = load_data(path_dev)
datafile_test = load_data(path_test)

print("Data loaded.")

print("Start to create the dataset ...")

dataset_train, vocab_words, vocab_phones = create_dataset(datafile_train, batch_size = config["batch_size"], return_vocab = True)

dataset_train, dataset_val = train_test_split(dataset_train, test_size = config["test_size"])

print("Dataset created.")


# --------------------------------------------------
#              Model
# --------------------------------------------------

print("Creation of the model ... ")

# ------ Create the model ------

vocab_size_words = len(vocab_words)
vocab_size_phoneme = len(vocab_phones)
embedding_dim_words = config["embedding_dim_words"]
embedding_dim_phones = config["embedding_dim_phones"]
rnn_units = config["rnn_units"]
nb_layers = config["nb_layer"]

model = MyModel(vocab_size_phoneme=vocab_size_phoneme, vocab_size_words=vocab_size_words, embedding_dim_words=embedding_dim_words, embedding_dim_phoneme=embedding_dim_phones, rnn_units = rnn_units, nb_layers=nb_layers) 

# Call the model, just to get the shapes
for input, true_output in dataset_train.take(1):
    true_output_text = true_output[0]
    true_output_phoneme = true_output[1]
    example_batch_predictions_words, example_batch_predictions_phoneme = model(input)


print("Here is the model:")
print(model.summary())


# ------ Train the model ------

if TRAINING:

    print("Start training the model...")

    model, history = train_model(model, dataset_train, dataset_val, epochs = config["epoch"], learning_rate=config["learning_rate"], save_model = SAVE_MODEL_PATH)

    print("Model trained.")

# ------ Load the model ------

else :

    print("Start loading the model...")

    model.load_weights(SAVE_MODEL_PATH)

    print("Model loaded.")



# --------------------------------------------------
#       Predict Brain with activation layers
#(Only for one subject and one record for the moment)
# --------------------------------------------------

SUBJECT = "01"
RECORD_NUMBER = "1"
    
print("Load the data containing the kinetic energy ...")

df_energy, df_annotations = load_data_meg(subject = SUBJECT, record_number = RECORD_NUMBER)

print("Data loaded.")

if GET_ACTIVATIONS :
    print("Start to create the dataset with the sentences at the different timesteps...")

    every_sentence_words, every_sentence_phonemes = get_every_phonemes_and_words(df_energy, df_annotations, vocab_words, vocab_phones)

    print("Dataset created.")   

    print("Get hidden and memory states of the model at each timestep ...")

    memory, hidden = collect_activations(model, every_sentence_words, every_sentence_phonemes) # Shapes: (n_layers, n_timesteps, rnn_size)

    print("Hidden and memory states collected for every timestep.")
    
    folder = os.path.join(SAVE_HIDDEN_MEMORY_FOLDER, SUBJECT, RECORD_NUMBER)
    if not os.path.exists(folder):
        os.makedirs(folder)

    save_memory_path = os.path.join(folder, "memory.npy")
    np.save(save_memory_path, memory)

    save_hidden_path = os.path.join(folder, "hidden.npy")
    np.save(save_hidden_path, hidden)

else :
    print("Load hidden and memory states ...")
    
    save_memory_path = os.path.join(SAVE_HIDDEN_MEMORY_FOLDER, SUBJECT, RECORD_NUMBER, "memory.npy")
    save_hidden_path = os.path.join(SAVE_HIDDEN_MEMORY_FOLDER, SUBJECT, RECORD_NUMBER, "hidden.npy")

    memory = np.load(save_memory_path)
    hidden = np.load(save_hidden_path)
    
    print("Activations loaded.")   

# Train - Test Split 
TEST_SIZE_TIMESTEP = 0.25

timestep_split = int(memory.shape[1] * TEST_SIZE_TIMESTEP)

memory_train, memory_test = memory[:, :timestep_split], memory[:, timestep_split:]
hidden_train, hidden_test = hidden[:, :timestep_split], hidden[:, timestep_split:]

cols_before_split, cols_after_split = df_energy.columns[:timestep_split], df_energy.columns[timestep_split:]
df_energy_train, df_energy_test = df_energy[cols_before_split], df_energy[cols_after_split]

print("Starting the regression ...")

energy_predict_from_memory, energy_predict_from_hidden, real_energy = regression(memory_train, hidden_train, memory_test, hidden_test, df_energy_train, df_energy_test)

print("Regression done.")