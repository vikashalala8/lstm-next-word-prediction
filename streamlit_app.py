import streamlit as st
import tensorflow as tf
import pickle
import numpy as np
import pandas as pd

from tensorflow.keras.preprocessing.sequence import pad_sequences


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="LSTM Next Word Prediction",
    page_icon="🧠",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

/* Main background */
.stApp {
    background-color: #f5f7fb;
}

/* Main title */
.main-title {
    font-size: 32px;
    font-weight: 700;
    color: #1f2937;
}

/* Subtitle */
.subtitle {
    color: #6b7280;
    font-size: 15px;
}

/* Cards */
.card {
    background-color: white;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #e5e7eb;
}

/* Prediction word */
.big-word {
    font-size: 38px;
    font-weight: 800;
    text-align: center;
}

/* Small text */
.small-text {
    color: #6b7280;
    text-align: center;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #111827;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    model = tf.keras.models.load_model(
        "model/next_word_model.keras"
    )

    return model


# ============================================================
# LOAD TOKENIZER
# ============================================================

@st.cache_resource
def load_tokenizer():

    with open(
        "model/tokenizer.pickle",
        "rb"
    ) as file:

        tokenizer = pickle.load(file)

    return tokenizer


model = load_model()
tokenizer = load_tokenizer()


# ============================================================
# MODEL INFORMATION
# ============================================================

max_sequence_len = model.input_shape[1] + 1

total_words = len(
    tokenizer.word_index
) + 1


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🧠 LSTM AI")

    st.write(
        "Next Word Prediction System"
    )

    st.divider()

    st.subheader("Dashboard")

    st.write("🏠 Prediction")
    st.write("📊 Prediction Analysis")
    st.write("⚙️ Model Information")

    st.divider()

    st.subheader("Model")

    st.write("Architecture: LSTM")

    st.write("Embedding: 32")

    st.write("LSTM Units: 64")

    st.write("Activation: Softmax")


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">'
    '🧠 LSTM Next Word Prediction Dashboard'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Natural Language Processing • Deep Learning • LSTM'
    '</div>',
    unsafe_allow_html=True
)

st.write("")


# ============================================================
# TOP KPI CARDS
# ============================================================

col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        label="📚 Vocabulary Size",
        value=total_words
    )


with col2:

    st.metric(
        label="🧠 LSTM Units",
        value="64"
    )


with col3:

    st.metric(
        label="🔢 Embedding Size",
        value="32"
    )


with col4:

    st.metric(
        label="🎯 Output",
        value="Softmax"
    )


st.write("")


# ============================================================
# INPUT SECTION
# ============================================================

st.subheader("✍️ Enter Text")

text = st.text_input(
    "Enter your sentence",
    placeholder="Example: I love machine",
    label_visibility="collapsed"
)


predict_button = st.button(
    "🔮 PREDICT NEXT WORD",
    use_container_width=True
)


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_next_words(
    seed_text,
    top_n=5
):

    # Convert words to numbers

    token_list = tokenizer.texts_to_sequences(
        [seed_text]
    )[0]


    # No known words

    if len(token_list) == 0:

        return []


    # Padding

    token_list = pad_sequences(
        [token_list],
        maxlen=max_sequence_len - 1,
        padding="pre"
    )


    # Model prediction

    prediction = model.predict(
        token_list,
        verbose=0
    )[0]


    # Get top indexes

    top_indices = np.argsort(
        prediction
    )[-top_n:][::-1]


    results = []


    for index in top_indices:

        if index == 0:

            continue


        word = None


        for w, word_index in tokenizer.word_index.items():

            if word_index == index:

                word = w

                break


        if word is not None:

            # IMPORTANT:
            # Convert NumPy float32
            # into normal Python float

            probability = float(
                prediction[index]
            ) * 100.0


            results.append(
                {
                    "Word": word,
                    "Probability": probability
                }
            )


    return results


# ============================================================
# PREDICTION
# ============================================================

if predict_button:

    if text.strip() == "":

        st.warning(
            "⚠️ Please enter some text."
        )


    else:

        results = predict_next_words(
            text,
            top_n=5
        )


        if not results:

            st.error(
                "❌ The entered words are not "
                "available in the training vocabulary."
            )


        else:

            # =================================================
            # MAIN RESULT
            # =================================================

            best_word = results[0]["Word"]

            best_probability = float(
                results[0]["Probability"]
            )


            st.write("")


            st.subheader(
                "🎯 Prediction Result"
            )


            result_col1, result_col2 = st.columns(
                2
            )


            with result_col1:

                st.markdown(
                    '<div class="card">',
                    unsafe_allow_html=True
                )

                st.markdown(
                    "<p style='text-align:center;"
                    "color:#6b7280;'>"
                    "Predicted Next Word"
                    "</p>",
                    unsafe_allow_html=True
                )

                st.markdown(
                    f"<div class='big-word'>"
                    f"{best_word}"
                    f"</div>",
                    unsafe_allow_html=True
                )

                st.markdown(
                    "</div>",
                    unsafe_allow_html=True
                )


            with result_col2:

                st.markdown(
                    '<div class="card">',
                    unsafe_allow_html=True
                )

                st.markdown(
                    "<p style='text-align:center;"
                    "color:#6b7280;'>"
                    "Confidence"
                    "</p>",
                    unsafe_allow_html=True
                )

                st.markdown(
                    f"<div class='big-word'>"
                    f"{best_probability:.2f}%"
                    f"</div>",
                    unsafe_allow_html=True
                )

                st.markdown(
                    "</div>",
                    unsafe_allow_html=True
                )


            # =================================================
            # TOP PREDICTIONS
            # =================================================

            st.write("")

            st.subheader(
                "📊 Top 5 Predictions"
            )


            # Create DataFrame

            df = pd.DataFrame(
                results
            )


            # -----------------------------------------------
            # Chart
            # -----------------------------------------------

            chart_col, table_col = st.columns(
                [2, 1]
            )


            with chart_col:

                st.bar_chart(
                    df.set_index("Word")[
                        "Probability"
                    ]
                )


            # -----------------------------------------------
            # Table
            # -----------------------------------------------

            with table_col:

                table_df = df.copy()

                table_df["Probability"] = (
                    table_df["Probability"]
                    .apply(
                        lambda x:
                        f"{x:.2f}%"
                    )
                )


                st.dataframe(
                    table_df,
                    hide_index=True,
                    use_container_width=True
                )


            # =================================================
            # PROBABILITY BARS
            # =================================================

            st.write("")

            st.subheader(
                "📈 Prediction Confidence"
            )


            for rank, item in enumerate(
                results,
                start=1
            ):

                word = item["Word"]

                probability = float(
                    item["Probability"]
                )


                col_word, col_percentage = st.columns(
                    [4, 1]
                )


                with col_word:

                    st.write(
                        f"**{rank}. {word}**"
                    )


                with col_percentage:

                    st.write(
                        f"**{probability:.2f}%**"
                    )


                # IMPORTANT:
                # st.progress accepts normal float

                progress_value = float(
                    probability / 100.0
                )


                # Keep between 0 and 1

                progress_value = max(
                    0.0,
                    min(
                        progress_value,
                        1.0
                    )
                )


                st.progress(
                    progress_value
                )


            # =================================================
            # GENERATED OUTPUT
            # =================================================

            st.write("")

            st.subheader(
                "📝 Generated Output"
            )


            output = text.strip() + " " + best_word


            st.info(
                output
            )


# ============================================================
# MODEL ARCHITECTURE
# ============================================================

st.write("")

st.subheader(
    "⚙️ Model Architecture"
)


architecture = st.columns(5)


with architecture[0]:

    st.info(
        "📝\n\n"
        "**Input Text**"
    )


with architecture[1]:

    st.info(
        "🔢\n\n"
        "**Tokenizer**"
    )


with architecture[2]:

    st.info(
        "🧩\n\n"
        "**Embedding**\n\n"
        "32"
    )


with architecture[3]:

    st.info(
        "🧠\n\n"
        "**LSTM**\n\n"
        "64 Units"
    )


with architecture[4]:

    st.info(
        "🎯\n\n"
        "**Softmax**"
    )


# ============================================================
# MODEL DETAILS
# ============================================================

st.write("")

st.subheader(
    "📋 Model Details"
)


details_col1, details_col2 = st.columns(
    2
)


with details_col1:

    st.markdown(
        """
        **Model Type**

        LSTM-based language model

        **Embedding Dimension**

        32

        **LSTM Units**

        64
        """
    )


with details_col2:

    st.markdown(
        """
        **Output Layer**

        Dense + Softmax

        **Prediction Type**

        Next-word prediction

        **Training**

        Supervised learning
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "🧠 LSTM Next Word Prediction • "
    "NLP & Deep Learning Project"
)