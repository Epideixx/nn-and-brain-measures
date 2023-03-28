import numpy as np
import tensorflow as tf

def load_data(path):
    """
    Load the data from the given path, which is assumed to be a npz file.
    """

    data = np.load(path)
    data.allow_pickle = True

    return data

def create_dataset(data, batch_size, shuffle=True, buffer_size=10000, return_vocab = True):
    """
    Create a dataset from the given data, which is assumed to be from npz file with the following items :
        - vocab_phones
        - vocab_words
        - data_words_last
        - data_words_curr
        - data_phones
        - data_words_next
    """

    # Get the vocabularies and add the special <break> token
    vocab_phones = np.concatenate((data["vocab_phones"], np.array(['<break>'])))
    vocab_words = data["vocab_words"]

    id_break_words = int(np.where(vocab_words == '<break>')[0])
    id_break_phones = int(np.where(vocab_phones == '<break>')[0])

    # Copy the usefull datasets, such we don't have to call it directly form the .npz files
    data_words_last = data["data_words_last"].copy()
    data_words_curr = data["data_words_curr"].copy()
    data_phones = data["data_phones"].copy()

    # Concatenate all the tokens
    dataset_words_curr = np.concatenate([data_words_last[i] + [id_break_words] for i in range(len(data_words_last))])
    dataset_words_next = np.concatenate([data_words_curr[i] + [id_break_words] for i in range(len(data_words_curr))])
    dataset_phones = np.concatenate([data_phones[i] + [id_break_phones] for i in range(len(data_phones))])

    # tensor slices slits the data in tensor of dimension 0 containing only the value
    dataset_words_curr = tf.data.Dataset.from_tensor_slices(dataset_words_curr)
    dataset_words_next = tf.data.Dataset.from_tensor_slices(dataset_words_next)
    dataset_phones = tf.data.Dataset.from_tensor_slices(dataset_phones)
    dataset = tf.data.Dataset.zip((dataset_words_curr, dataset_words_next, dataset_phones))

    # Split the sentences in sequences of tokens
    seq_length = 100
    sequences = dataset.batch(seq_length+1, drop_remainder=True)

    # From those sequences, we create the input and the output of the model
    def split_input_target(sequence_words_last, sequence_words_curr,phonemes):
        input_text_words = sequence_words_last[:-1]
        input_text_phonemes = phonemes[:-1]
        target_text_words  = sequence_words_curr[1:] 
        target_text_phonemes = phonemes[1:]

        input_text = input_text_words, input_text_phonemes
        target_text = target_text_words, target_text_phonemes

        return input_text, target_text

    dataset = sequences.map(split_input_target)
    # Elements in the dataset are tuples (input_text, target_text)
    # input_text is a tensor of shape (seq_length, 2)
    # target_text is a tensor of shape (seq_length, 2)

    # Final dataset, in batches
    if shuffle :
        dataset = (
            dataset
            .shuffle(buffer_size)
            .batch(batch_size, drop_remainder=True))
        
    else :
        dataset = (
            dataset
           .batch(batch_size, drop_remainder=True))
    if return_vocab :
        return dataset, vocab_words, vocab_phones
    else :
        return dataset


def train_test_split(dataset, test_size=0.2):

    train = int((1 - test_size)* len(dataset))
    dataset_train, dataset_test = dataset.take(train), dataset.skip(train)

    return dataset_train, dataset_test