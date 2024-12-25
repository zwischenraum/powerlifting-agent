from typing import List, Dict
import json
from pathlib import Path
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models

class RulesSearch:
    def __init__(self, rules_file: str = "data/ipf_rules.json"):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.rules_file = Path(rules_file)
        self.rules_data = self._load_rules()
        self.rules_text = [rule['text'] for rule in self.rules_data]
        
        # Create BM25 index
        tokenized_rules = [text.split() for text in self.rules_text]
        self.bm25 = BM25Okapi(tokenized_rules)
        
        # Initialize Qdrant client
        self.qdrant = QdrantClient(host="localhost", port=6333)
        
        # Create collection if it doesn't exist
        self._init_collection()
        
        # Upload vectors if collection is empty
        self._upload_vectors()

    def _load_rules(self) -> List[Dict]:
        """Load rules from JSON file."""
        if not self.rules_file.exists():
            raise FileNotFoundError(f"Rules file not found: {self.rules_file}")
        with open(self.rules_file) as f:
            return json.load(f)

    def _init_collection(self):
        """Initialize Qdrant collection"""
        try:
            self.qdrant.get_collection('rules')
        except:
            self.qdrant.create_collection(
                collection_name='rules',
                vectors_config=models.VectorParams(
                    size=384,  # MiniLM-L6-v2 embedding size
                    distance=models.Distance.COSINE
                )
            )
    
    def _upload_vectors(self):
        """Upload vectors to Qdrant if collection is empty"""
        if self.qdrant.get_collection('rules').vectors_count == 0:
            embeddings = self.model.encode(self.rules_text)
            
            points = []
            for i, (embedding, rule) in enumerate(zip(embeddings, self.rules_data)):
                points.append(models.PointStruct(
                    id=i,
                    vector=embedding.tolist(),
                    payload={'text': rule['text']}
                ))
            
            self.qdrant.upload_points(
                collection_name='rules',
                points=points
            )

    def search(self, query: str, k: int = 3, alpha: float = 0.5) -> List[Dict]:
        """
        Perform hybrid search using both BM25 and semantic search with Qdrant.
        
        Args:
            query: Search query
            k: Number of results to return
            alpha: Weight for combining scores (0.5 means equal weight)
            
        Returns:
            List of top k matching rules with scores
        """
        # BM25 scoring
        bm25_scores = self.bm25.get_scores(query.split())
        bm25_scores = (bm25_scores - np.min(bm25_scores)) / (np.max(bm25_scores) - np.min(bm25_scores))
        
        # Semantic search scoring using Qdrant
        query_embedding = self.model.encode(query)
        semantic_results = self.qdrant.search(
            collection_name='rules',
            query_vector=query_embedding,
            limit=len(self.rules_text)  # Get all scores for hybrid ranking
        )
        
        # Create semantic scores array
        semantic_scores = np.zeros(len(self.rules_text))
        for hit in semantic_results:
            semantic_scores[hit.id] = hit.score
        
        # Combine scores
        combined_scores = alpha * semantic_scores + (1 - alpha) * bm25_scores
        
        # Get top k results
        top_k_idx = np.argsort(combined_scores)[-k:][::-1]
        
        results = []
        for idx in top_k_idx:
            results.append({
                'rule': self.rules_data[idx],
                'score': float(combined_scores[idx])
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
