import os

from pymilvus import MilvusClient


class PhotoRepository(object):

    def __init__(self, collection: str = "photo", vector_field: str = "embedding"):
        self.url = os.getenv('MILVUS_URL', 'http://localhost:19530')
        self.token = os.getenv('MILVUS_TOKEN', 'root:Milvus')

        self.client = MilvusClient(
            uri=self.url,
            token=self.token
        )

        self.collection = collection
        self.vector_field = vector_field

    def index(self, data):
        self.client.insert(
            collection_name=self.collection,
            data=data
        )

    def delete(self, ids: list[str]):
        deleted = self.client.delete(
            collection_name=self.collection,
            ids=ids
        )

        return deleted

    def search(self, data, limit: int = 10, out=None, params=None):
        if params is None:
            params = {"metric_type": "COSINE"}
        if out is None:
            out = ["pk"]

        return self.client.search(
            collection_name=self.collection,
            data=[data],
            anns_field=self.vector_field,
            search_params=params,
            limit=limit,
            output_fields=out
        )