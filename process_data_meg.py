import os
import hdf5storage
import pandas as pd
import numpy as np
import re

from sklearn.linear_model import RidgeCV
import tensorflow as tf

from tqdm import tqdm

def load_data_meg(main_folder_path = "data/MEG", subject = '01', record_number = '1'):
    """
    Loads the data from the MEG/Xiaobo's measures for a givem subject (01 -> 11) and a given record (1 -> 7).
    The returned data will be a dataframe containing the kinetic energy (rows = voxels and cols = timesteps), and a datframe with the corresponding annoted script.
    """


    # Load the energy file
    FILE_ENERGY = os.path.join(main_folder_path, "Subject" + subject, "Brain_activity_results" + record_number + ".mat")
    mat = hdf5storage.loadmat(FILE_ENERGY)

    # Load the data containing the timesteps
    FILE_TIME = os.path.join(main_folder_path, "Subject" + subject, "time_in_out_diff" + record_number + ".mat")
    mat_time = hdf5storage.loadmat(FILE_TIME)


    # Load the file containing the link between the TedTalk and the MEG recordings
    FILE_INFO_ANNOTATIONS= os.path.join(main_folder_path, "tedlium_annotations", "tedlium_meg_info.csv")
    info_file_annotations = pd.read_csv(FILE_INFO_ANNOTATIONS, index_col=0)

    # Find the name of the file containing the annotations we want (like the good talk)
    to_match = 'Tedlium' + subject + "/@raw" + ".*0" + record_number + "$"
    line_nb = [i for i in range(len(info_file_annotations)) if len(re.findall(to_match, info_file_annotations.iloc[i]['studypath'])) > 0][0]
    name_file_annotations = info_file_annotations.iloc[line_nb]["annotation"]

    # Load the file containing the annotations
    FILE_ANNOTATIONS = os.path.join(main_folder_path, "tedlium_annotations", name_file_annotations)
    df_annotations = pd.read_csv(FILE_ANNOTATIONS, index_col=0)


    # Create the dataframe containing the MEG/Xiaobo measure. Columns are the timesteps and the rows are the different voxels.
    energy = mat["Brain_activity_results"]["kinetic_diff_features"][0, 0][0][1]
    time = mat_time["time_in_out_diff"][0, 0][1][0]

    time = time[1:] #Because one timestep too much (check if it's the good one to remove)

    df_energy = pd.DataFrame(energy, columns=time)

    # Finally, remove all the timesteps before the speech actually begins
    cols_to_keep = [col for col in df_energy.columns if col >= min(df_annotations["minTime"])]
    df_energy = df_energy[cols_to_keep]

    return df_energy, df_annotations


def get_every_phonemes_and_words(df_energy, df_annotations, vocab_words, vocab_phones):
    """
    Returns a numpy array containing all the sequences of phonemes and words, as id token (so int)>
    Exemple, if the speech is : "The Neuro lab is amazing", we will get, translated in words but in real given as id tokens:
        every_sentence_words = array([The], [The, Neuro], [The, Neuro, lab], [The, Neuro, lab, is) ...
        every_sentence_phonemes = ... (similar)
    Each occurence corresponds to one timestep in the MEG signal (so you can have multiple times in a row the same phoneme or word)
    """

    # Final output
    every_sentence_words = []
    every_sentence_phonemes = []

    # Current sentence, in which we add at each timestep the current phoneme and the current word
    sentence_words = []
    sentence_phonemes = []

    for timestep in df_energy.columns:
        
        # If the MEG recording has started but not the speech we just skip
        if min(df_annotations["minTime"]) >= timestep:
            continue

        # We try get the phoneme + word happening during the timestep, else we take the word + phoneme right before the timestep
        row = df_annotations[df_annotations["minTime"] <= timestep][df_annotations["maxTime"] >= timestep]
        if len(row) == 0: # No words at this timestep, so we take the phoneme and words which occured just before
            row = df_annotations[df_annotations["maxTime"] <= timestep]
            phoneme, word = str(row.iloc[-1]["phone"]), str(row.iloc[-1]["word"])
        else:
            phoneme, word = str(row.iloc[0]["phone"]), str(row.iloc[0]["word"])

        # We tokenize the phonemes and words, and take into account if the word is unknown
        if word not in vocab_words:
            word = "<unk>"    
        word_token = int(np.where(vocab_words == word)[0])
        phoneme_token = int(np.where(vocab_phones == phoneme)[0])

        # Finally we just add everything
        sentence_words.append(word_token)
        sentence_phonemes.append(phoneme_token)

        every_sentence_words.append(sentence_words.copy())
        every_sentence_phonemes.append(sentence_phonemes.copy())

    # Transform in Numpy array, as it is more efficient
    every_sentence_words = np.array(every_sentence_words)
    every_sentence_phonemes = np.array(every_sentence_phonemes) 

    return every_sentence_words, every_sentence_phonemes


def collect_activations(model, every_sentence_words, every_sentence_phonemes, return_proba_correct = False):
    """
    At each timestep, we collect the hidden activation and the memory activation of the model.
    """

    memory = []
    hidden = []

    proba_correct_word = []
    proba_correct_phoneme = []

    for i in tqdm(range(len(every_sentence_words)), total = len(every_sentence_words)):

        pred_words, pred_phonemes, memory_every_layer, states_every_layer = model((np.array(every_sentence_words[i])[np.newaxis, ...], np.array(every_sentence_phonemes[i])[np.newaxis, ...]), return_state=True)

        memory.append(memory_every_layer)
        hidden.append(states_every_layer)

        pred_words = tf.squeeze(pred_words, axis=0)
        pred_phonemes = tf.squeeze(pred_phonemes, axis=0)

        if i < len(every_sentence_words) - 1:
            proba_correct_word.append(pred_words[-1, every_sentence_words[i+1][-1]])
            proba_correct_phoneme.append(pred_phonemes[-1, every_sentence_phonemes[i+1][-1]])

        else : 
            proba_correct_word.append(pred_words[-1, -1])
            proba_correct_phoneme.append(pred_phonemes[-1, -1])

    memory = np.moveaxis(np.concatenate(memory, axis=0), [0], [1]) # Shape: (n_layers, n_timesteps, rnn_size)
    hidden = np.moveaxis(np.concatenate(hidden, axis=0), [0], [1]) # Shape: (n_layers, n_timesteps, rnn_size)

    if return_proba_correct:
        return memory, hidden, proba_correct_word, proba_correct_phoneme
    else:
        return memory, hidden, 


def regression(memory_train, hidden_train, memory_test, hidden_test, df_energy_train, df_energy_test, shift = 0):
    """
    Compute a regression model to predict every voxels' kinetic energy from the hidden and memory activations separatly.
    """
    energy_train = np.moveaxis(df_energy_train.to_numpy(), [0], [1]) # Shape: (n_timesteps, n_voxels)
    energy_test= np.moveaxis(df_energy_test.to_numpy(), [0], [1]) # Shape: (n_timesteps, n_voxels)

    energy_train = energy_train[shift:]
    energy_test = energy_test[shift:]
    
    energy_predict_from_memory = np.zeros((memory_test.shape[0], energy_test.shape[0], energy_test.shape[1]))
    energy_predict_from_hidden = np.zeros((hidden_test.shape[0], energy_test.shape[0], energy_test.shape[1]))

    for i in range(len(memory_train)):
        activation_train = memory_train[i].copy() # Shape: (n_timesteps, rnn_size)
        activation_test = memory_test[i].copy() # Shape: (n_timesteps, rnn_size)

        activation_train = activation_train[:-shift]
        activation_test = activation_test[:-shift]

        model = RidgeCV()
        model.fit(activation_train, energy_train)

        energy_predict = model.predict(activation_test)

        energy_predict_from_memory[i] = energy_predict


    for i in range(len(hidden_train)):
        activation_train = hidden_train[i].copy() # Shape: (n_timesteps, rnn_size)
        activation_test = hidden_test[i].copy() # Shape: (n_timesteps, rnn_size)

        activation_train = activation_train[:-shift]
        activation_test = activation_test[:-shift]

        model = RidgeCV()
        model.fit(activation_train, energy_train)

        energy_predict = model.predict(activation_test)

        energy_predict_from_hidden[i] = energy_predict

    real_energy = energy_test.copy()

    return energy_predict_from_memory, energy_predict_from_hidden, real_energy

    



