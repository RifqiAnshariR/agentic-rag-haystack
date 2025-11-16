import streamlit as st
from haystack.components.agents import Agent
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.utils import Secret
from haystack.dataclasses import ChatMessage
from haystack_experimental.chat_message_stores.in_memory import InMemoryChatMessageStore
from haystack_experimental.components.writers import ChatMessageWriter
from src.database import get_product_document_store, get_common_info_document_store
from src.pipeline import ParaphraserPipeline, MetaDataFilterPipeline, ProductRAGPipeline, CommonInfoPipeline, ChatHistoryPipeline
from src.tools import get_product_tool, get_common_info_tool
from config.config import Config
from utils.styling import load_css


load_css()


def response_handler(query):
    st.session_state.chat_message_writer.run([ChatMessage.from_user(query)])
    
    history_context = st.session_state.chat_history_pipeline.run()
    messages = [
        ChatMessage.from_system(f"Conversation Context: {history_context}"),
        ChatMessage.from_user(query)
    ]

    response = st.session_state.agent.run(messages=messages)
    response_text = response["messages"][-1].text

    st.session_state.chat_message_writer.run([ChatMessage.from_assistant(response_text)])
    return response_text


def setup_pipelines():
    if "chat_message_store" not in st.session_state:
        st.session_state.chat_message_store = InMemoryChatMessageStore()

    if "chat_message_writer" not in st.session_state:
        st.session_state.chat_message_writer = ChatMessageWriter(st.session_state.chat_message_store)

    if "product_store" not in st.session_state:
        st.session_state.product_store = get_product_document_store()

    if "common_info_store" not in st.session_state:
        st.session_state.common_info_store = get_common_info_document_store()

    if "paraphraser" not in st.session_state:
        st.session_state.paraphraser = ParaphraserPipeline(st.session_state.chat_message_store)

    if "metadata_filter" not in st.session_state:
        st.session_state.metadata_filter = MetaDataFilterPipeline()

    if "product_rag" not in st.session_state:
        st.session_state.product_rag = ProductRAGPipeline(st.session_state.product_store)

    if "common_info_rag" not in st.session_state:
        st.session_state.common_info_rag = CommonInfoPipeline(st.session_state.common_info_store)

    if "chat_history_pipeline" not in st.session_state:
        st.session_state.chat_history_pipeline = ChatHistoryPipeline(st.session_state.chat_message_store)

    if "agent" not in st.session_state:
        product_tool = get_product_tool(
            st.session_state.paraphraser,
            st.session_state.metadata_filter,
            st.session_state.product_rag
        )
        common_info_tool = get_common_info_tool(
            st.session_state.paraphraser,
            st.session_state.common_info_rag
        )

        st.session_state.agent = Agent(
            chat_generator=OpenAIChatGenerator(
                model="gpt-4.1-2025-04-14",
                api_key=Secret.from_token(Config.OPENAI_API_KEY)
            ),
            tools=[product_tool, common_info_tool],
            system_prompt="""
            You are a helpful shop assistant.
            ROUTING LOGIC:
            1. If user asks about specific PRODUCTS (price, material, recommendation), use 'retrieve_and_generate_recommendation'.
            2. If user asks about GENERAL INFO (shipping, refund, payment, how to buy), use 'common_info_tool'.
            3. If user says hello or general chit-chat, reply directly.
            """,
            max_agent_steps=10,
        )
        st.session_state.agent.warm_up()


if __name__ == "__main__":
    st.set_page_config(page_icon=Config.PAGE_ICON, layout="wide")

    st.html("<h1>Smart Shopper Assistant</h1>")
    
    setup_pipelines()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        role = message["role"]
        content = message["content"]
        css_class = "user-bubble" if role == "user" else "assistant-bubble"
        st.html(f"<div class='chat-bubble {css_class}'>{content}</div>")

    if prompt := st.chat_input("Hello, what can I help you today?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.html(f"<div class='chat-bubble user-bubble'>{prompt}</div>")

        with st.spinner("Generating a response..."):
            response = response_handler(prompt)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.html(f"<div class='chat-bubble assistant-bubble'>{response}</div>")

    st.html("<div class='footer'>©2025 Rifqi Anshari Rasyid.</div>")
