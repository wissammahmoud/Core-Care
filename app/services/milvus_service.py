from pymilvus import connections, Collection
from sentence_transformers import SentenceTransformer
import numpy as np
from threading import Lock
import logging
import torch

logger = logging.getLogger(__name__)

class MilvusManager:
    """Singleton Milvus connection manager"""
    _instance = None
    _lock = Lock()
    
    def __init__(self):
        self.collection = None
        self.embedding_model = None
        self.cache = {}  # Embedding cache
        self.macro_cache = {}  # Macro cache
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance
    
    def connect(self, host: str, port: int, collection_name: str):
        """Connect to Milvus and load embedding model"""
        if self.collection is not None:
            return self.collection
        
        with self._lock:
            if self.collection is not None:
                return self.collection
            
            logger.info(f"Connecting to Milvus at {host}:{port}")
            
            connections.connect(
                alias="default",
                host=host,
                port=port
            )
            
            self.collection = Collection(collection_name)
            self.collection.load()
            
            logger.info("Loading embedding model...")
            self.embedding_model = SentenceTransformer(
                'alexdseo/RecipeBERT',
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
            
            logger.info("Milvus connected successfully")
            return self.collection
    
    def ensure_connection(self):
        """Ensure Milvus connection is alive"""
        try:
            self.collection.num_entities
        except Exception as e:
            logger.warning(f"Connection lost, reconnecting... {e}")
            # Reconnect logic here
    
    def embed_text(self, texts: list) -> list:
        """Batch embed texts with caching"""
        vectors = []
        to_compute = []
        indices = []
        
        for i, text in enumerate(texts):
            key = text.strip().lower()
            if key in self.cache:
                vectors.append(self.cache[key])
            else:
                to_compute.append(text)
                indices.append(i)
        
        if to_compute:
            embeddings = self.embedding_model.encode(
                to_compute,
                normalize_embeddings=True
            )
            for idx, emb in zip(indices, embeddings):
                vec = emb.astype(np.float32).tolist()
                self.cache[texts[idx].strip().lower()] = vec
                vectors.insert(idx, vec)
        
        return vectors
    
    def search_ingredients(self, ingredient_names: list, top_k: int = 3):
        """Batch search for ingredients"""
        self.ensure_connection()
        
        # Check cache first
        cached_results = []
        uncached = []
        
        for name in ingredient_names:
            if name in self.macro_cache:
                cached_results.append(self.macro_cache[name])
            else:
                uncached.append(name)
        
        if not uncached:
            return [{"query": name, "results": [self.macro_cache[name]]} 
                    for name in ingredient_names]
        
        # Embed uncached items
        vectors = self.embed_text(uncached)
        
        # Batch search
        results = self.collection.search(
            data=vectors,
            anns_field="item_name_emb",
            param={"metric_type": "COSINE", "params": {"ef": 80}},
            limit=top_k,
            output_fields=["item_name", "protein_g", "fat_g", "carb_g"],
        )
        
        # Process results
        output = []
        for query_name, hits in zip(uncached, results):
            hits_list = []
            for rank, hit in enumerate(hits, 1):
                entity = hit.entity
                item = {
                    "rank": rank,
                    "score": float(hit.distance),
                    "item_name": entity.get("item_name"),
                    "protein_g": float(entity.get("protein_g", 0.0)),
                    "fat_g": float(entity.get("fat_g", 0.0)),
                    "carb_g": float(entity.get("carb_g", 0.0)),
                }
                item["calories"] = (
                    4.0 * item["protein_g"] + 
                    4.0 * item["carb_g"] + 
                    9.0 * item["fat_g"]
                )
                hits_list.append(item)
                
                # Cache best match
                if rank == 1:
                    self.macro_cache[query_name] = item
            
            output.append({"query": query_name, "results": hits_list})
        
        return output