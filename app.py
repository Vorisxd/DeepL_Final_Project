import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from rag import query_and_generate_response
from loguru import logger
import warnings

warnings.filterwarnings("ignore")

logger.add("app.log", rotation="1 MB")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.set_page_config(page_title="AI Lawyer", page_icon="⚖️")

st.title("AI Lawyer")

for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.markdown(message.content)
    else:
        with st.chat_message("AI"):
            st.markdown(message.content)

user_query = st.chat_input("Your message")

if user_query is not None and user_query != "":
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    logger.info(f"Human message: {user_query}")

    with st.chat_message("Human"):
        st.markdown(user_query)

    with st.chat_message("AI"):
        ai_response = query_and_generate_response(user_query, st.session_state.chat_history, stream_response=True)
        st.write(ai_response)
        logger.info(f"AI response: {ai_response}")

    st.session_state.chat_history.append(AIMessage(content=ai_response))
