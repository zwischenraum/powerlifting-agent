from typing import List, Dict
import logging
from pathlib import Path
import numpy as np
from rank_bm25 import BM25Okapi
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams
from openai import OpenAI

class RulesSearch:
    """A hybrid search engine for powerlifting rules combining BM25 and semantic search.
    
    This class implements a search system that combines lexical search using BM25
    with semantic search using Qdrant vector database. Results are combined using
    Reciprocal Rank Fusion (RRF) for optimal ranking.
    """
    
    # Constant for RRF calculation (typical value is 60)
    RRF_C = 60
    
    def __init__(self, rules_file: str = "data/rulebook.txt", openai_client: OpenAI = None):
        """Initialize the search engine.
        
        Args:
            rules_file: Path to the text file containing powerlifting rules
        """
        self.rules_file = Path(rules_file)
        self.rules_chunks = self._load_and_chunk_rules()
        
        # Create BM25 index
        tokenized_rules = [text.split() for text in self.rules_chunks]
        self.bm25 = BM25Okapi(tokenized_rules)
        
        # Initialize clients
        self.qdrant = QdrantClient(host="localhost", port=6333)
        self.openai = openai_client or OpenAI()
        
        # Create collection if it doesn't exist
        self._init_collection()
        
        # Upload texts if collection is empty
        self._upload_texts()

    def _load_and_chunk_rules(self) -> List[str]:
        """Load rules from text file and split into semantic chunks."""
        if not self.rules_file.exists():
            raise FileNotFoundError(f"Rules file not found: {self.rules_file}")
            
        with open(self.rules_file) as f:
            text = f.read()
            
        # Split into paragraphs first
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        # Combine short paragraphs and split long ones to get reasonable chunk sizes
        chunks = []
        current_chunk = []
        current_length = 0
        
        for para in paragraphs:
            para_words = len(para.split())
            
            # If paragraph is very long, split it into sentences
            if para_words > 100:
                sentences = [s.strip() for s in para.split('.') if s.strip()]
                for sentence in sentences:
                    if current_length + len(sentence.split()) > 100:
                        if current_chunk:
                            chunks.append(' '.join(current_chunk))
                            current_chunk = []
                            current_length = 0
                    current_chunk.append(sentence)
                    current_length += len(sentence.split())
            else:
                if current_length + para_words > 100:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = []
                    current_length = 0
                current_chunk.append(para)
                current_length += para_words
                
        if current_chunk:
            chunks.append(' '.join(current_chunk))
            
        return chunks

    def _init_collection(self):
        """Initialize Qdrant collection with text indexing"""
        collection_name = 'rules'
        try:
            # Try to get the collection
            collection_info = self.qdrant.get_collection(collection_name)
            logging.info(f"Collection exists with {collection_info.points_count} vectors")
        except Exception as e:
            logging.info(f"Creating new collection: {collection_name}")
            self.qdrant.recreate_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=1536,  # Size for OpenAI embeddings
                    distance=Distance.COSINE
                )
            )
            # Get updated collection info
            collection_info = self.qdrant.get_collection(collection_name)
            logging.info(f"Created new collection with {collection_info.points_count} vectors")
    
    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for text using OpenAI API."""
        response = self.openai.embeddings.create(
            model="text-embedding-ada-002",
            input=text
        )
        return response.data[0].embedding

    def _upload_texts(self):
        """Upload texts with embeddings to Qdrant if collection is empty"""
        collection_info = self.qdrant.get_collection('rules')
        
        if collection_info.points_count == 0:
            logging.debug(f"Getting embeddings for {len(self.rules_chunks)} text chunks...")
            
            # Get embeddings for all chunks at once
            response = self.openai.embeddings.create(
                model="text-embedding-ada-002",
                input=self.rules_chunks
            )
            embeddings = [item.embedding for item in response.data]
            
            logging.debug("Creating points...")
            points = [
                models.PointStruct(
                    id=i,
                    vector=embedding,
                    payload={'text': chunk}
                )
                for i, (chunk, embedding) in enumerate(zip(self.rules_chunks, embeddings))
            ]
            
            logging.debug("Uploading points to Qdrant...")
            try:
                self.qdrant.upload_points(
                    collection_name='rules',
                    points=points,
                    wait=True
                )
                logging.info("Successfully uploaded all points")
            except Exception as e:
                logging.error(f"Error uploading points: {str(e)}")
                raise
        else:
            logging.info(f"Collection already contains {collection_info.points_count} vectors")

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
        
        # Get semantic search rankings using embeddings
        query_vector = self._get_embedding(query)
        semantic_results = self.qdrant.search(
            collection_name='rules',
            query_vector=query_vector,
            limit=len(self.rules_chunks)
        )
        
        # Convert semantic results to ranks
        semantic_ranks = np.zeros(len(self.rules_chunks), dtype=int)
        for rank, hit in enumerate(semantic_results):
            semantic_ranks[hit.id] = rank
            
        # Calculate RRF scores
        rrf_scores = np.zeros(len(self.rules_chunks))
        for idx in range(len(self.rules_chunks)):
            rrf_scores[idx] = (1 / (self.RRF_C + bm25_ranks[idx])) + \
                             (1 / (self.RRF_C + semantic_ranks[idx]))
        
        # Get top k results
        top_k_idx = np.argsort(rrf_scores)[-k:][::-1]
        
        results = []
        for idx in top_k_idx:
            # Get individual scores
            bm25_score = float(bm25_scores[idx])
            semantic_score = 1.0 - float(semantic_ranks[idx] / len(self.rules_chunks))  # Convert rank to similarity
            rrf_score = float(rrf_scores[idx])
            
            results.append({
                'text': self.rules_chunks[idx],
                'bm25_score': bm25_score,
                'semantic_score': semantic_score, 
                'rrf_score': rrf_score
            })
            
        return results

def search_rules(query: str, openai_client: OpenAI = None) -> str:
    """Function to be used by the Rules Agent."""
    try:
        searcher = RulesSearch(openai_client=openai_client)
        results = searcher.search(query)
        
        # Format results as a readable string
        output = "Here are the most relevant rules:\n\n"
        for i, result in enumerate(results, 1):
            output += f"{i}. {result['text']}\n"
            output += f"   BM25 Score: {result['bm25_score']:.3f}\n"
            output += f"   Semantic Score: {result['semantic_score']:.3f}\n"
            output += f"   RRF Score: {result['rrf_score']:.3f}\n\n"
        
        return output
    except Exception as e:
        return f"Error searching rules: {str(e)}"
