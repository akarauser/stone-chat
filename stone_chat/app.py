import ollama
import streamlit as st

from .utils._logger import logger
from .utils._validation import config_args

# Streamlit app title
st.set_page_config(page_title="Stone Chat")

# Session States
if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "chosen_model" not in st.session_state:
    st.session_state["chosen_model"] = ""

if "model_set" not in st.session_state:
    st.session_state["model_set"] = set()


# Function to get model description
def get_description(model_name):
    """
    Retrieve the description of model.
    """
    try:
        return config_args.descriptions[model_name]
    except Exception as e:
        logger.error(f"Error getting model description: {e}")
        return "Description not available."


def handle_conversation():
    """
    Streams the responses from the Ollama chat API.
    """
    try:
        stream = ollama.chat(
            model=st.session_state.chosen_model,
            messages=st.session_state.messages,
            stream=True,
        )
        for chunk in stream:
            yield chunk["message"]["content"]

    except Exception as e:
        logger.error(f"Error during chat stream: {e}")
        yield "Error generating response."


def conversation():
    """
    Displays the conversation history in the chat interface.
    """
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if prompt := st.chat_input("Type here..."):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            message = st.write_stream(handle_conversation())
            st.session_state.messages.append({"role": "assistant", "content": message})


# Model Selection
try:
    try:
        local_models_list = sorted(
            [
                ollama.list()["models"][i]["model"]
                for i in range(len(ollama.list()["models"]))
            ]
        )
        local_models_list = [
            model for model in local_models_list if "embed" not in model
        ]

        st.session_state.chosen_model = st.sidebar.selectbox(
            "Local models list",
            options=local_models_list,
            index=None,
            placeholder="",
            help="Choose a model from the list.",
        )
    except Exception as e:
        logger.error(f"Error retrieving local models: {e}")
        st.error("Failed to load local models.  Please check your Ollama installation.")

    # Main Logic
    if not st.session_state.chosen_model:
        st.session_state.model_set.clear()

    elif st.session_state.chosen_model not in st.session_state.model_set:
        try:
            st.session_state.model_set.clear()
            st.session_state.model_set.add(st.session_state.chosen_model)
            st.sidebar.write(get_description(st.session_state.chosen_model))
            st.session_state.messages = []
            conversation()
        except Exception as e:
            logger.error(f"Failed the initialize new model: {e}")

    else:
        try:
            st.sidebar.write(get_description(st.session_state.chosen_model))
            conversation()
        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")

except Exception as e:
    logger.error(f"Ollama connection failed: {e}")
    st.error(f"Ollama connection failed: {e}")
    st.write(
        "You can run the desktop application from your computer or `ollama serve` command can be used when you want to start ollama without running the desktop application."
    )
