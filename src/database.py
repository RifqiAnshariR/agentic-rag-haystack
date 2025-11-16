from haystack_integrations.document_stores.mongodb_atlas import MongoDBAtlasDocumentStore
from haystack import component
from pymongo import MongoClient
from typing import List
from config.config import Config


class MongoDBConnection:
    def __init__(self):
        self.connection_string = Config.MONGO_CONNECTION_STRING
        self.client = MongoClient(self.connection_string)
        self.db = self.client.depato_store

    def get_materials(self):
        return [doc['name'] for doc in self.db.materials.find()]

    def get_categories(self):
        return [doc['name'] for doc in self.db.categories.find()]


def get_product_document_store():
    return MongoDBAtlasDocumentStore(
        database_name="depato_store",
        collection_name="products",
        vector_search_index="vector_index",
        full_text_search_index="search_index",
    )


def get_common_info_document_store():
    return MongoDBAtlasDocumentStore(
        database_name="depato_store",
        collection_name="common_info",
        vector_search_index="vector_index",
        full_text_search_index=None,
    )


@component
class GetMaterials:
    def __init__(self):
        self.db = MongoDBConnection()

    @component.output_types(materials=List[str])
    def run(self):
        return {"materials": self.db.get_materials()}


@component
class GetCategories:
    def __init__(self):
        self.db = MongoDBConnection()

    @component.output_types(categories=List[str])
    def run(self):
        return {"categories": self.db.get_categories()}
