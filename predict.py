import numpy as np
import tensorflow as tf
import pickle

from tensorflow.keras.preprocessing.sequence import pad_sequences


# --------------------------------
# 1. Load trained model
# --------------------------------

model = tf.keras.models.load_model(
    "model/next_word_model.keras"
)


# --------------------------------
# 2. Load tokenizer
# --------------------------------

with open(
    "model/tokenizer.pickle",
    "rb"
) as file:

    tokenizer = pickle.load(file)


# --------------------------------
# 3. Maximum sequence length
# --------------------------------

max_sequence_len = model.input_shape[1] + 1


# --------------------------------
# 4. Prediction function
# --------------------------------

def predict_next_word(seed_text):

    # Convert input words into numbers
    token_list = tokenizer.texts_to_sequences(
        [seed_text]
    )[0]

    # Make sequence the required length
    token_list = pad_sequences(
        [token_list],
        maxlen=max_sequence_len - 1,
        padding="pre"
    )

    # Predict
    predicted = model.predict(
        token_list,
        verbose=0
    )

    # Get word index with highest probability
    predicted_word_index = np.argmax(
        predicted,
        axis=1
    )[0]

    # Convert number back to word
    for word, index in tokenizer.word_index.items():

        if index == predicted_word_index:
            return word

    return None


# --------------------------------
# 5. Test
# --------------------------------

print("Prediction 1:")
print(predict_next_word("I love"))

print("\nPrediction 2:")
print(predict_next_word("I enjoy"))

print("\nPrediction 3:")
print(predict_next_word("deep"))

print("\nPrediction 4:")
print(predict_next_word("machine"))

print("\nPrediction 5:")
print(predict_next_word("I love machine"))