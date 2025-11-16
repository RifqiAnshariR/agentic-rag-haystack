import re
import json
from haystack.tools.tool import Tool
from functools import partial

def product_rag_logic(query: str, paraphraser, metadata_filter, rag_pipeline):
    paraphrased_query = paraphraser.run(query)
    filter_result = metadata_filter.run(paraphrased_query)
    
    filters = {}
    try:
        json_match = re.search(r'```json\n(.*?)\n```', filter_result, re.DOTALL)
        if json_match:
            filters = json.loads(json_match.group(1))
        else:
            filters = json.loads(filter_result)
    except:
        filters = {}

    return rag_pipeline.run(paraphrased_query, filters)

def common_info_logic(query: str, paraphraser, rag_pipeline):
    paraphrased_query = paraphraser.run(query)
    return rag_pipeline.run(paraphrased_query)

def get_product_tool(paraphraser, metadata_filter, rag_pipeline):
    return Tool(
        name="retrieve_and_generate_recommendation",
        description="Retrieve products based on user query using metadata filtering and vector search.",
        function=partial(product_rag_logic, paraphraser=paraphraser, metadata_filter=metadata_filter, rag_pipeline=rag_pipeline),
        parameters={
            "type": "object",
            "properties": {"query": {"type": "string", "description": "User query about products."}},
            "required": ["query"]
        }
    )

def get_common_info_tool(paraphraser, rag_pipeline):
    return Tool(
        name="common_info_tool",
        description="Answer general questions about shipping, payment, refund, and how to buy.",
        function=partial(common_info_logic, paraphraser=paraphraser, rag_pipeline=rag_pipeline),
        parameters={
            "type": "object",
            "properties": {"query": {"type": "string", "description": "User question about general info."}},
            "required": ["query"]
        }
    )
