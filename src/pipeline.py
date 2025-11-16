import os
from haystack import Pipeline
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.generators import OpenAIGenerator
from haystack.components.builders import ChatPromptBuilder, PromptBuilder
from haystack.dataclasses import ChatMessage
from haystack_experimental.components.retrievers import ChatMessageRetriever
from haystack_integrations.components.retrievers.mongodb_atlas import MongoDBAtlasEmbeddingRetriever
from haystack.utils import Secret
from src.database import GetMaterials, GetCategories
from config.config import Config


class ParaphraserPipeline:
    def __init__(self, chat_message_store):
        self.pipeline = Pipeline()
        self.pipeline.add_component(
            "memory_retriever", 
            ChatMessageRetriever(chat_message_store)
        )
        self.pipeline.add_component(
            "prompt_builder", 
            ChatPromptBuilder(variables=["query", "memories"])
        )
        self.pipeline.add_component(
            "generator",
            OpenAIChatGenerator(
                model="gpt-4.1",
                api_key=Secret.from_token(Config.OPENAI_API_KEY)
            ),
        )
        self.pipeline.connect("memory_retriever", "prompt_builder.memories")
        self.pipeline.connect("prompt_builder.prompt", "generator.messages")

    def run(self, query):
        messages = [
            ChatMessage.from_system(
                "You are a helpful assistant that paraphrases user queries based on conversation history to make them standalone."
            ),
            ChatMessage.from_user(
                """
                History:
                {% for memory in memories %}{{memory.text}}{% endfor %}
                Query: {{query}}
                Paraphrased Query:
                """
            ),
        ]
        res = self.pipeline.run(
            data={"prompt_builder": {"query": query, "template": messages}},
            include_outputs_from=["generator"]
        )
        return res["generator"]["replies"][0].text


class MetaDataFilterPipeline:
    def __init__(self):
        self.pipeline = Pipeline()
        self.pipeline.add_component(
            "materials", 
            GetMaterials()
        )
        self.pipeline.add_component(
            "categories",
            GetCategories()
        )
        self.pipeline.add_component(
            "prompt_builder",
            PromptBuilder(template=Config.PRODUCTS_METADATA_FILTER_TEMPLATE)
        )
        self.pipeline.add_component(
            "generator",
            OpenAIGenerator(
                model="gpt-4.1",
                api_key=Secret.from_token(Config.OPENAI_API_KEY)
            ),
        )
        self.pipeline.connect("materials.materials", "prompt_builder.materials")
        self.pipeline.connect("categories.categories", "prompt_builder.categories")
        self.pipeline.connect("prompt_builder", "generator")

    def run(self, query):
        res = self.pipeline.run(data={"prompt_builder": {"input": query}})
        return res["generator"]["replies"][0]


class ProductRAGPipeline:
    def __init__(self, document_store):
        self.pipeline = Pipeline()
        self.pipeline.add_component(
            "embedder",
            SentenceTransformersTextEmbedder(model="sentence-transformers/all-mpnet-base-v2"),
        )
        self.pipeline.add_component(
            "retriever",
            MongoDBAtlasEmbeddingRetriever(document_store=document_store, top_k=10),
        )
        self.pipeline.add_component(
            "prompt_builder", 
            ChatPromptBuilder(variables=["query", "documents"])
        )
        self.pipeline.add_component(
            "generator",
            OpenAIChatGenerator(
                model="gpt-4.1",
                api_key=Secret.from_token(Config.OPENAI_API_KEY)
            ),
        )
        self.pipeline.connect("embedder", "retriever")
        self.pipeline.connect("retriever", "prompt_builder.documents")
        self.pipeline.connect("prompt_builder.prompt", "generator.messages")

    def run(self, query, filter_dict):
        messages = [
            ChatMessage.from_system("You are a helpful shop assistant giving product recommendations."),
            ChatMessage.from_user(
                """
                Query: {{query}}
                Products Found:
                {% for product in documents %}
                {{loop.index}}. Name: {{ product.meta.title }}, Price: {{ product.meta.price }}, Material: {{ product.meta.material }}, Category: {{ product.meta.category }}
                Content: {{ product.content}}
                {% endfor %}
                Answer:
                """
            ),
        ]
        res = self.pipeline.run(
            data={
                "embedder": {"text": query},
                "retriever": {"filters": filter_dict},
                "prompt_builder": {"query": query, "template": messages},
            }
        )
        return res["generator"]["replies"][0].text


class CommonInfoPipeline:
    def __init__(self, document_store):
        self.pipeline = Pipeline()
        self.pipeline.add_component(
            "embedder",
            SentenceTransformersTextEmbedder(model="sentence-transformers/all-mpnet-base-v2"),
        )
        self.pipeline.add_component(
            "retriever",
            MongoDBAtlasEmbeddingRetriever(document_store=document_store, top_k=5),
        )
        self.pipeline.add_component(
            "prompt_builder", 
            ChatPromptBuilder(variables=["query", "documents"])
        )
        self.pipeline.add_component(
            "generator",
            OpenAIChatGenerator(
                model="gpt-4.1",
                api_key=Secret.from_token(Config.OPENAI_API_KEY)
            ),
        )
        self.pipeline.connect("embedder", "retriever")
        self.pipeline.connect("retriever", "prompt_builder.documents")
        self.pipeline.connect("prompt_builder.prompt", "generator.messages")

    def run(self, query):
        messages = [
            ChatMessage.from_system("You are a helpful shop assistant. Answer based on the documents."),
            ChatMessage.from_user(
                """
                Query: {{query}}
                Information:
                {% for doc in documents %}
                Topic: {{ doc.meta['topic'] }}
                Content: {{ doc.content }}
                {% endfor %}
                Answer:
                """
            ),
        ]
        res = self.pipeline.run(
            data={
                "embedder": {"text": query},
                "prompt_builder": {"query": query, "template": messages},
            }
        )
        return res["generator"]["replies"][0].text


class ChatHistoryPipeline:
    def __init__(self, chat_message_store):
        self.pipeline = Pipeline()
        self.pipeline.add_component(
            "memory_retriever",
            ChatMessageRetriever(chat_message_store)
        )
        self.pipeline.add_component(
            "prompt_builder",
            PromptBuilder(template="{% for memory in memories %}{{memory.text}}\n{% endfor %}")
        )

        self.pipeline.connect("memory_retriever", "prompt_builder.memories")

    def run(self):
        res = self.pipeline.run(data={}, include_outputs_from=["prompt_builder"])
        return res["prompt_builder"]["prompt"]
