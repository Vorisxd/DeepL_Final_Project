import os
from vec_store_setup import initialize_vectorstore
from dotenv import load_dotenv
from prompts import context_template, fin_resp_template
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from loguru import logger
import warnings

warnings.filterwarnings("ignore")

load_dotenv()
huggingface_api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
langchain_api_key = os.getenv("LANGCHAIN_API_KEY")
langchain_project_name = os.getenv("LANGCHAIN_PROJECT")
langchain_endpoint = os.getenv("LANGCHAIN_ENDPOINT")
langchain_tracing = os.getenv("LANGCHAIN_TRACING_V2")

retriever = initialize_vectorstore()
retriever = retriever.as_retriever(search_kwargs={"k": 5})


gen_llm = HuggingFaceEndpoint(
    huggingfacehub_api_token=huggingface_api_token, 
    repo_id="Qwen/Qwen2.5-72B-Instruct",
    max_new_tokens=124,
    do_sample=True,
    temperature=0.2,
    repetition_penalty=1.5,
    )

context_llm = HuggingFaceEndpoint(
    huggingfacehub_api_token=huggingface_api_token, 
    repo_id="google/gemma-2-9b-it",
    max_new_tokens=124,
    do_sample=True,
    temperature=0.1,
    repetition_penalty=1.5,
    )


def contextualize_query(user_query, chat_history):
    if len(chat_history) > 10:
        chat_history = chat_history[-10:]


    context_prompt = ChatPromptTemplate.from_template(context_template)

    context_query_chain = (context_prompt 
                           | context_llm 
                           | StrOutputParser()
                            )

    query = context_query_chain.invoke(
        {"chat_history": chat_history, "user_question": user_query}
    )
    return query

def query_and_generate_response(user_query, chat_history, stream_response=False):
    
    fin_prompt = ChatPromptTemplate.from_template(fin_resp_template)

    fin_rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | fin_prompt
        | gen_llm
        | StrOutputParser()
    )

    standalone_query = contextualize_query(
        user_query=user_query, chat_history=chat_history
    )
    logger.info(f"Standalone query: {standalone_query}")
    if stream_response:
        response = fin_rag_chain.stream(standalone_query)
        logger.info(f"Generated response: {response}")
    else:
        response = fin_rag_chain.invoke(standalone_query)

    logger.info(f"Generated response: {response}")
    return response


