from typing import List, Dict
import json
from pathlib import Path
import numpy as np
from rank_bm25 import BM25Okapi
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, TextConfig, VectorParams

class RulesSearch:
    """A hybrid search engine for powerlifting rules combining BM25 and semantic search.
    
    This class implements a search system that combines lexical search using BM25
    with semantic search using Qdrant vector database. Results are combined using
    Reciprocal Rank Fusion (RRF) for optimal ranking.
    """
    
    # Constant for RRF calculation (typical value is 60)
    RRF_C = 60
    
    def __init__(self, rules_file: str = "data/ipf_rules.json"):
        """Initialize the search engine.
        
        Args:
            rules_file: Path to the JSON file containing powerlifting rules
        """
        self.rules_file = Path(rules_file)
        self.rules_data = self._load_rules()
        self.rules_text = [rule['text'] for rule in self.rules_data]
        
        # Create BM25 index
        tokenized_rules = [text.split() for text in self.rules_text]
        self.bm25 = BM25Okapi(tokenized_rules)
        
        # Initialize Qdrant client with text encoder
        self.qdrant = QdrantClient(host="localhost", port=6333)
        
        # Create collection if it doesn't exist
        self._init_collection()
        
        # Upload texts if collection is empty
        self._upload_texts()

    def _load_rules(self) -> List[Dict]:
        """Load rules from JSON file."""
        if not self.rules_file.exists():
            raise FileNotFoundError(f"Rules file not found: {self.rules_file}")
        with open(self.rules_file) as f:
            return json.load(f)

    def _init_collection(self):
        """Initialize Qdrant collection with text indexing"""
        try:
            self.qdrant.get_collection('rules')
        except:
            self.qdrant.create_collection(
                collection_name='rules',
                vectors_config=VectorParams(
                    size=384,  # Default size for onnx text encoder
                    distance=Distance.COSINE
                ),
                sparse_vectors_config={
                    "text": TextConfig(
                        tokenizer=TextConfig.Tokenizer(
                            type="word",
                            lowercase=True,
                            min_token_len=2,
                            max_token_len=20,
                        )
                    )
                }
            )
    
    def _upload_texts(self):
        """Upload texts to Qdrant if collection is empty"""
        if self.qdrant.get_collection('rules').vectors_count == 0:
            points = []
            for i, rule in enumerate(self.rules_data):
                points.append(models.PointStruct(
                    id=i,
                    vector=rule,
                    payload={
                        'text': rule['text'],
                    }
                ))
            
            self.qdrant.upload_points(
                collection_name='rules',
                points=points
            )

    def search(self, query: str, k: int = 3) -> List[Dict]:
        """Perform hybrid search using both BM25 and semantic search with Qdrant.
        
        This method combines BM25 and semantic search results using Reciprocal Rank 
        Fusion (RRF). RRF is a robust method for combining multiple ranked lists 
        without requiring score normalization.
        
        Args:
            query: Search query string
            k: Number of top results to return
            
        Returns:
            List of dictionaries containing top k matching rules with scores
        """
        # Get BM25 rankings
        bm25_scores = self.bm25.get_scores(query.split())
        bm25_ranks = (-bm25_scores).argsort()
        
        # Get semantic search rankings
        semantic_results = self.qdrant.search(
            collection_name='rules',
            query_text=query,  # Using text search instead of vector
            limit=len(self.rules_text),
            query_filter=None
        )
        
        # Convert semantic results to ranks
        semantic_ranks = np.zeros(len(self.rules_text), dtype=int)
        for rank, hit in enumerate(semantic_results):
            semantic_ranks[hit.id] = rank
            
        # Calculate RRF scores
        rrf_scores = np.zeros(len(self.rules_text))
        for idx in range(len(self.rules_text)):
            rrf_scores[idx] = (1 / (self.RRF_C + bm25_ranks[idx])) + \
                             (1 / (self.RRF_C + semantic_ranks[idx]))
        
        # Get top k results
        top_k_idx = np.argsort(rrf_scores)[-k:][::-1]
        
        results = []
        for idx in top_k_idx:
            results.append({
                'rule': self.rules_data[idx],
                'score': float(rrf_scores[idx])
            })
            
        return results

def search_rules(query: str) -> str:
    """Function to be used by the Rules Agent."""
    try:
        searcher = RulesSearch()
        results = searcher.search(query)
        
        # Format results as a readable string
        output = "Here are the most relevant rules:\n\n"
        for i, result in enumerate(results, 1):
            output += f"{i}. {result['rule']['text']}\n"
            output += f"   (Score: {result['score']:.3f})\n\n"
        
        return output
    except Exception as e:
        return f"Error searching rules: {str(e)}"
