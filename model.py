# ------------------------------------
#           Dependencies
# ------------------------------------

import tensorflow as tf
import numpy as np
import os

from wandb.keras import WandbMetricsLogger
# ------------------------------------
#          Model
# ------------------------------------

class MyModel(tf.keras.Model):
  
  def __init__(self, vocab_size_words, vocab_size_phoneme, embedding_dim_words, embedding_dim_phoneme, rnn_units, nb_layers = 1, dropout_rate = 0.1):
    
    super().__init__(self)

    # Embedding layers 
    self.embedding_words = tf.keras.layers.Embedding(vocab_size_words, embedding_dim_words)
    self.embedding_phoneme = tf.keras.layers.Embedding(vocab_size_phoneme, embedding_dim_phoneme)

    # LSTM layers
    self.lstm = [tf.keras.layers.LSTM(rnn_units,
                                   return_sequences=True,
                                   return_state=True) for _ in range(nb_layers)]
    
    # Final prediction layers
    self.dense_words = tf.keras.layers.Dense(vocab_size_words, activation="softmax")
    self.dense_phoneme = tf.keras.layers.Dense(vocab_size_phoneme, activation="softmax")


  def call(self, inputs, return_state=False, training=False):

    # List to store the activations of each layer
    states_every_layer = []
    memory_every_layer = []

    words, phonemes = inputs

    # Embedding
    words = self.embedding_words(words, training=training)
    phonemes = self.embedding_phoneme(phonemes, training=training)
    x = tf.concat([words, phonemes], axis=-1)

    # Pass through LSTM + save states
    for lstm in self.lstm:
        x, memory, states = lstm(x, training=training)
        memory_every_layer.append(memory[..., np.newaxis, :])
        states_every_layer.append(states[..., np.newaxis, :])

    # Save the current state of the LSTM
    memory_every_layer = tf.concat(memory_every_layer, axis=1)
    states_every_layer = tf.concat(states_every_layer, axis=1)

    # Final prediction
    pred_words = self.dense_words(x, training=training)
    pred_phoneme = self.dense_phoneme(x, training=training)
    
    if return_state:
      return pred_words, pred_phoneme, memory_every_layer, states_every_layer
      # Shapes : (batch_size, seq_len, vocab_words_size), (batch_size, seq_len, vocab_phoneme_size), (batch_size, nb_layers, rnn_units), (batch_size, nb_layers, rnn_units)
    else:
      return pred_words, pred_phoneme
      # Shapes : (batch_size, seq_len, vocab_words_size)


def train_model(model, data_train, data_val, epochs, learning_rate, checkpoint_path = './training_checkpoints', save_model = "my_model"):

  loss = tf.losses.SparseCategoricalCrossentropy(from_logits=False)
  lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    learning_rate,
    decay_steps=1000,
    decay_rate=0.96,
    staircase=True)
  optimizer = tf.optimizers.Adam(learning_rate=lr_schedule)

  # Name of the checkpoint files
  checkpoint_prefix = os.path.join(checkpoint_path, "check_point_{epoch}")

  checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
      filepath=checkpoint_prefix,
      save_weights_only=True)

  model.compile(optimizer=optimizer, loss=loss)

  # Train the model
  history = model.fit(data_train, epochs=epochs, callbacks=[checkpoint_callback, WandbMetricsLogger(log_freq=1)], validation_data=data_val)

  # Save the model
  model.save_weights(save_model)
  
  return model, history


def predict_model(model, words, phones, vocab_phones, vocab_words, n_steps_to_predict = 1):

  """
  Predicts the phonemes and words for a given sequence of words and phonemes, given as lists.
  It will predict the n_steps_to_predict following words and phonemes.
  """
  words_to_return = words.copy()
  phones_to_return = phones.copy()

  def recursive_predict(words, phones, n_steps_to_predict):

    # Tokenization
    token_phones = []
    token_words = []

    for word, phoneme in zip(words, phones):

      token_phone = int(np.where(vocab_phones == phoneme)[0])
      token_word = np.where(vocab_words == word)
      if len(token_words) == 0 :
          token_word = int(np.where(vocab_words == "<unk>")[0])
      else :
        token_word = int(token_words[0])

      token_phones.append(token_phone)
      token_words.append(token_word)

    # Transform input to put in the model
    token_words = tf.expand_dims(np.array(token_words), axis=0)
    token_phones = tf.expand_dims(np.array(token_phones), axis=0)

    # Predict the following word and phoneme
    pred_words, pred_phonemes = model((token_words, token_phones))

    # Transform output back to words and phonemes
    pred_words = tf.squeeze(pred_words, axis=0)
    pred_phonemes = tf.squeeze(pred_phonemes, axis=0)

    pred_words = tf.argmax(pred_words, axis=-1)
    pred_phonemes = tf.argmax(pred_phonemes, axis=-1)

    pred_word = vocab_words[pred_words[-1]]
    pred_phoneme = vocab_phones[pred_phonemes[-1]]

    words_to_return.append(pred_word)
    phones_to_return.append(pred_phoneme)

    if n_steps_to_predict == 1:
      return words_to_return, phones_to_return
    
    else :
      return recursive_predict(words_to_return, phones_to_return, n_steps_to_predict - 1)
    
  
  return recursive_predict(words_to_return, phones_to_return, n_steps_to_predict)
