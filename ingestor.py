import os
import pandas as pd
from pymongo import MongoClient
from dotenv import load_dotenv
from haystack import Pipeline
from haystack.dataclasses import Document
from haystack.document_stores.types import DuplicatePolicy
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.writers import DocumentWriter
from haystack_integrations.document_stores.mongodb_atlas import MongoDBAtlasDocumentStore
from config.config import Config


load_dotenv()


def ingest_products():
    df = pd.read_pickle(Config.DATASET_PRODUCTS_FILE)

    documents = []
    for _, row in df.iterrows():
        descriptions = row["description"].strip("[]").strip("''")
        document = Document(
            content = f"{row['title']}\n {descriptions}",
            meta = {
                'asin': row['asin'],
                'title': row['title'],
                'brand': row['brand'],
                'price': row['price'],
                'gender': row['gender'],
                'material': row['material'],
                'category': row['category'],
            }
        )
        documents.append(document)

    document_store = MongoDBAtlasDocumentStore(
        database_name="depato_store",
        collection_name="products",
        vector_search_index="vector_index",
        full_text_search_index="search_index",
    )
        
    pipeline = Pipeline()
    pipeline.add_component("embedder",SentenceTransformersDocumentEmbedder(
        model="sentence-transformers/all-mpnet-base-v2"     # 768
    ))
    pipeline.add_component("writer",DocumentWriter(
        document_store=document_store,
        policy=DuplicatePolicy.OVERWRITE
    ))
    pipeline.connect("embedder","writer")

    pipeline.run({"embedder": {"documents": documents}})


def ingest_materials_categories():
    df = pd.read_pickle(Config.DATASET_FILE)
    
    materials = df['material'].unique().tolist()
    categories = df['category'].unique().tolist()
    
    client = MongoClient(os.environ['MONGO_CONNECTION_STRING'])
    db = client.depato_store

    material_collection = db.materials
    category_collection = db.categories

    documents_material= [{"name":m} for m in materials]
    documents_category = [{"name":c} for c in categories]

    material_collection.insert_many(documents_material)
    category_collection.insert_many(documents_category)


def ingest_common_info():
    df = pd.read_csv(Config.DATASET_COMMON_INFO_FILE)

    documents = []
    for _, row in df.iterrows():
        documents.append(Document(content=row["content"], meta={"topic": row["topic"]}))

    document_store = MongoDBAtlasDocumentStore(
        database_name="depato_store",
        collection_name="common_info",
        vector_search_index="vector_index",
        full_text_search_index=None
    )

    pipeline = Pipeline()
    pipeline.add_component("embedder", SentenceTransformersDocumentEmbedder(
        model="sentence-transformers/all-mpnet-base-v2"     # 768
    ))
    pipeline.add_component("writer", DocumentWriter(document_store))
    pipeline.connect("embedder", "writer")

    pipeline.run({"embedder": {"documents": documents}})


if __name__ == "__main__":
    ingest_products()
    ingest_materials_categories()
    ingest_common_info()
