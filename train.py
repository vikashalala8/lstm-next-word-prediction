import numpy as np
import tensorflow as tf
import pickle

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense


# --------------------------------
# 1. Read training text
# --------------------------------

with open("data/text.txt", "r", encoding="utf-8") as file:
    text = file.read()


# --------------------------------
# 2. Tokenization
# --------------------------------

tokenizer = Tokenizer()

tokenizer.fit_on_texts([text])

word_index = tokenizer.word_index

print("Word Index:")
print(word_index)


# --------------------------------
# 3. Create input sequences
# --------------------------------

sequences = []

for line in text.strip().split("\n"):

    # Convert words into numbers
    token_list = tokenizer.texts_to_sequences([line])[0]

    # Create n-gram sequences
    for i in range(1, len(token_list)):

        n_gram_sequence = token_list[:i + 1]

        sequences.append(n_gram_sequence)


print("\nSequences:")
print(sequences)


# --------------------------------
# 4. Padding
# --------------------------------

max_sequence_len = max(
    [len(seq) for seq in sequences]
)

print("\nMaximum sequence length:")
print(max_sequence_len)


input_sequences = np.array(
    pad_sequences(
        sequences,
        maxlen=max_sequence_len,
        padding="pre"
    )
)

print("\nPadded sequences:")
print(input_sequences)


# --------------------------------
# 5. Create X and y
# --------------------------------

X = input_sequences[:, :-1]

y = input_sequences[:, -1]

print("\nX:")
print(X)

print("\ny:")
print(y)


# --------------------------------
# 6. Number of words
# --------------------------------

total_words = len(tokenizer.word_index) + 1

print("\nTotal words:")
print(total_words)


# --------------------------------
# 7. Build LSTM model
# --------------------------------

model = Sequential()

model.add(
    Embedding(
        total_words,
        32,
        input_length=max_sequence_len - 1
    )
)

model.add(
    LSTM(64)
)

model.add(
    Dense(
        total_words,
        activation="softmax"
    )
)


# Display model
model.summary()


# --------------------------------
# 8. Compile model
# --------------------------------

model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)


# --------------------------------
# 9. Train model
# --------------------------------

history = model.fit(
    X,
    y,
    epochs=100,
    verbose=1
)


# --------------------------------
# 10. Save model
# --------------------------------

model.save(
    "model/next_word_model.keras"
)


# --------------------------------
# 11. Save tokenizer
# --------------------------------

with open(
    "model/tokenizer.pickle",
    "wb"
) as file:

    pickle.dump(tokenizer, file)


print("\nTraining completed!")

print("Model saved as:")
print("model/next_word_model.keras")

print("\nTokenizer saved as:")
print("model/tokenizer.pickle")