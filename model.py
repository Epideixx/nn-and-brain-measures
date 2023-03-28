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
  optimizer = tf.optimizers.Adam(learning_rate=learning_rate)

  # Name of the checkpoint files
  checkpoint_prefix = os.path.join(checkpoint_path, "check_point_{epoch}")

  checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
      filepath=checkpoint_prefix,
      save_weights_only=True)

  model.compile(optimizer=optimizer, loss=loss)

  history = model.fit(data_train, epochs=epochs, callbacks=[checkpoint_callback, WandbMetricsLogger(log_freq=1)], validation_data=data_val)

  model.save(save_model)
  
  return model, history