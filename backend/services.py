import json
import pickle
import os
from datetime import datetime
from typing import List, Optional, Tuple
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from .models import Report, QueryRequest, QueryResponse
from .config import settings

class EmbeddingService:
    def __init__(self):
        self.model = None
        self.use_openai = False
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize embedding model (OpenAI or sentence-transformers)"""
        if settings.OPENAI_API_KEY:
            try:
                from openai import OpenAI
                self.openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)
                self.use_openai = True
                print("Using OpenAI embeddings")
            except ImportError:
                print("OpenAI not available, falling back to sentence-transformers")
                self._load_sentence_transformer()
            except Exception as e:
                print(f"OpenAI setup failed ({e}), falling back to sentence-transformers")
                self._load_sentence_transformer()
        else:
            self._load_sentence_transformer()
    
    def _load_sentence_transformer(self):
        """Load sentence-transformers model"""
        self.model = SentenceTransformer(settings.EMBEDDING_MODEL)
        self.use_openai = False
        print(f"Using sentence-transformers model: {settings.EMBEDDING_MODEL}")
    
    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts"""
        if self.use_openai:
            return self._get_openai_embeddings(texts)
        else:
            return self._get_sentence_transformer_embeddings(texts)
    
    def _get_openai_embeddings(self, texts: List[str]) -> np.ndarray:
        """Get embeddings using OpenAI API"""
        embeddings = []
        for text in texts:
            response = self.openai_client.embeddings.create(
                input=text,
                model="text-embedding-ada-002"
            )
            embeddings.append(response.data[0].embedding)
        return np.array(embeddings)
    
    def _get_sentence_transformer_embeddings(self, texts: List[str]) -> np.ndarray:
        """Get embeddings using sentence-transformers"""
        return self.model.encode(texts, convert_to_numpy=True)

class VectorSearchService:
    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.index = None
        self.reports = []
        self.report_lookup = {}
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        self.tfidf_matrix = None
        
    def load_reports(self, reports_path: str):
        """Load reports from JSON file"""
        with open(reports_path, 'r') as f:
            data = json.load(f)
        
        self.reports = [Report(**report) for report in data]
        self.report_lookup = {report.id: report for report in self.reports}
        print(f"Loaded {len(self.reports)} reports")
        
        # Create embeddings and build index
        self._build_vector_index()
        self._build_tfidf_index()
    
    def _build_vector_index(self):
        """Build FAISS vector index from report texts"""
        print("Building vector index...")
        texts = [report.text for report in self.reports]
        embeddings = self.embedding_service.get_embeddings(texts)
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)  # L2 distance instead of inner product
        
        # Add embeddings directly without normalization to avoid FAISS compatibility issues
        self.index.add(embeddings.astype('float32'))
        
        # Save index and reports for persistence
        os.makedirs("data", exist_ok=True)
        faiss.write_index(self.index, settings.VECTOR_INDEX_PATH)
        with open(settings.REPORTS_INDEX_PATH, 'wb') as f:
            pickle.dump((self.reports, self.report_lookup), f)
        
        print(f"Vector index built with {self.index.ntotal} vectors")
    
    def _build_tfidf_index(self):
        """Build TF-IDF index for keyword matching"""
        texts = [report.text for report in self.reports]
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
        print("TF-IDF index built")
    
    def load_existing_index(self):
        """Load existing vector index and reports"""
        try:
            self.index = faiss.read_index(settings.VECTOR_INDEX_PATH)
            with open(settings.REPORTS_INDEX_PATH, 'rb') as f:
                self.reports, self.report_lookup = pickle.load(f)
            
            # Rebuild TF-IDF index (lightweight)
            self._build_tfidf_index()
            print(f"Loaded existing index with {len(self.reports)} reports")
            return True
        except (FileNotFoundError, Exception) as e:
            print(f"Could not load existing index: {e}")
            return False
    
    def search(self, query: str, site_id: Optional[str] = None, 
               date_range: Optional[List[str]] = None, top_k: int = None) -> List[Report]:
        """Search for relevant reports using semantic similarity and filters"""
        if top_k is None:
            top_k = settings.TOP_K_RESULTS
        
        # Get semantic similarity scores
        query_embedding = self.embedding_service.get_embeddings([query])
        
        # Search in vector index (convert to float32 for FAISS compatibility)
        scores, indices = self.index.search(query_embedding.astype('float32'), min(len(self.reports), top_k * 3))
        
        # Get candidate reports
        candidates = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.reports):  # Valid index
                report = self.reports[idx]
                candidates.append((report, float(score)))
        
        # Apply filters
        filtered_candidates = []
        for report, score in candidates:
            # Site filter
            if site_id and report.siteId != site_id:
                continue
            
            # Date range filter
            if date_range and len(date_range) == 2:
                report_date = datetime.fromisoformat(report.date.replace('Z', '+00:00'))
                start_date = datetime.fromisoformat(date_range[0].replace('Z', '+00:00'))
                end_date = datetime.fromisoformat(date_range[1].replace('Z', '+00:00'))
                
                if not (start_date <= report_date <= end_date):
                    continue
            
            filtered_candidates.append((report, score))
        
        # Sort by relevance score and return top k
        filtered_candidates.sort(key=lambda x: x[1], reverse=True)
        return [report for report, _ in filtered_candidates[:top_k]]

class ConversationalLLMService:
    def __init__(self):
        self.use_openai = False
        self._initialize_llm()
    
    def _initialize_llm(self):
        """Initialize LLM for conversational responses"""
        if settings.OPENAI_API_KEY:
            try:
                from openai import OpenAI
                # Test the API key by creating a client
                OpenAI(api_key=settings.OPENAI_API_KEY)
                self.use_openai = True
                print("LLM: Using OpenAI GPT for conversational responses")
            except ImportError:
                print("LLM: OpenAI not available, responses will be limited")
            except Exception as e:
                print(f"LLM: OpenAI setup failed ({e}), responses will be limited")
        else:
            print("LLM: No OpenAI key provided, responses will be limited")
    
    def generate_conversational_response(self, query: str, reports: List[Report]) -> str:
        """Generate a conversational response to the user's query using LLM only"""
        if not reports:
            return self._generate_no_results_response(query)
        
        if self.use_openai:
            return self._generate_openai_response(query, reports)
        else:
            # Fallback if OpenAI is not available - still try to provide a basic response
            # but encourage using OpenAI for better results
            current_date = datetime.now().strftime("%Y-%m-%d")
            return f"Found {len(reports)} relevant reports for your query '{query}' (current date: {current_date}, data available: Aug 25 - Sep 3, 2025). However, for detailed analysis and conversational responses, please configure OpenAI API key for better results."
    
    def _generate_no_results_response(self, query: str) -> str:
        """Generate response when no reports are found using LLM"""
        if self.use_openai:
            try:
                from openai import OpenAI
                client = OpenAI(api_key=settings.OPENAI_API_KEY)
                
                # Get current date for context
                current_date = datetime.now().strftime("%Y-%m-%d")
                
                prompt = f"""You are Guard Owl, a helpful security assistant. Speak naturally and conversationally, like you're talking to a colleague.

CURRENT DATE: {current_date}
AVAILABLE DATA: Security reports from August 25, 2025 to September 3, 2025

A user searched for: "{query}" but no matching security reports were found.

Respond conversationally and helpfully:
- NO formal greetings or email structure
- Be direct but friendly
- Acknowledge no results were found
- Suggest 2-3 specific alternatives or ask follow-up questions
- If they asked about dates outside our range, suggest our available timeframe
- Keep it natural and brief (2-3 sentences max)

Examples of good responses:
- "I didn't find any tailgating incidents, but there were several suspicious vehicle reports. Want me to check those instead?"
- "No incidents from last week since our data covers Aug 25 - Sep 3. Should I search that timeframe instead?"
- "Nothing at S06, but I see activity at S01-S05. Want to check those sites?"

Just respond naturally:"""

                messages = [
                    {"role": "system", "content": "You are Guard Owl, a helpful security assistant. Always speak naturally and conversationally like you're talking to a colleague. Never use formal language, greetings, or email-style responses. Be direct, friendly, and helpful."},
                    {"role": "user", "content": prompt}
                ]
                
                response = client.chat.completions.create(
                    model="gpt-4o-2024-05-13",
                    messages=messages,
                    max_tokens=150,
                    temperature=0.8
                )
                
                return response.choices[0].message.content.strip()
                
            except Exception as e:
                print(f"OpenAI API error for no-results response: {e}")
                # Simple fallback
                current_date = datetime.now().strftime("%Y-%m-%d")
                return f"I couldn't find any security reports matching '{query}' (current date: {current_date}, data available: Aug 25 - Sep 3, 2025). Try different keywords or adjust your search filters."
        else:
            current_date = datetime.now().strftime("%Y-%m-%d")
            return f"No reports found for '{query}' (current date: {current_date}, data available: Aug 25 - Sep 3, 2025). Please configure OpenAI API key for enhanced search assistance."
    
    def _generate_openai_response(self, query: str, reports: List[Report]) -> str:
        """Generate response using OpenAI GPT"""
        try:
            from openai import OpenAI
            client = OpenAI(api_key=settings.OPENAI_API_KEY)
            
            # Prepare context from reports
            context = self._prepare_context_for_llm(reports)
            
            # Get current date for context
            current_date = datetime.now().strftime("%Y-%m-%d")
            
            prompt = f"""You are Guard Owl, a knowledgeable security assistant helping analyze reports. You speak naturally and conversationally, like a helpful colleague.

CURRENT DATE: {current_date}
AVAILABLE DATA: Security reports from August 25, 2025 to September 3, 2025

A user asked: "{query}"

Based on these security reports:
{context}

Respond naturally and conversationally. Rules:
- NO formal greetings like "Hello" or sign-offs like "Best regards"
- NO email-style structure
- Talk like you're having a conversation with a colleague
- Be direct and helpful
- Keep it concise (2-3 paragraphs max)
- If dates are outside our data range, mention it naturally
- Use casual but professional language

Just answer their question directly and conversationally:"""

            messages = [
                {"role": "system", "content": "You are Guard Owl, a knowledgeable security assistant. Always respond in a natural, conversational tone like you're talking to a colleague. Never use formal email language, greetings, or sign-offs. Be direct, helpful, and casual but professional."},
                {"role": "user", "content": prompt}
            ]
            
            response = client.chat.completions.create(
                model="gpt-4o-2024-05-13",
                messages=messages,
                max_tokens=300,
                temperature=0.8
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"OpenAI API error: {e}")
            # Simple fallback when OpenAI fails
            current_date = datetime.now().strftime("%Y-%m-%d")
            return f"Found {len(reports)} relevant security reports for your query '{query}' (current date: {current_date}, data available: Aug 25 - Sep 3, 2025), but I'm unable to provide detailed analysis due to an API error. Please try again or check your OpenAI configuration."
    
    
    def _prepare_context_for_llm(self, reports: List[Report]) -> str:
        """Prepare report context for LLM"""
        context_parts = []
        for i, report in enumerate(reports[:3], 1):  # Limit to top 3 for token efficiency
            context_parts.append(f"{i}. {report.siteId} ({report.date[:10]}): {report.text}")
        return "\n".join(context_parts)

class ReportAnalysisService:
    def __init__(self):
        self.vector_service = VectorSearchService()
        self.llm_service = ConversationalLLMService()
        
    def initialize(self, reports_path: str):
        """Initialize the service with reports data"""
        # Try to load existing index first
        if not self.vector_service.load_existing_index():
            # Build new index from reports
            self.vector_service.load_reports(reports_path)
    
    def query(self, request: QueryRequest) -> QueryResponse:
        """Process a query and return structured response"""
        # Search for relevant reports
        relevant_reports = self.vector_service.search(
            query=request.query,
            site_id=request.siteId,
            date_range=request.dateRange
        )
        
        # Generate conversational summary using LLM
        if relevant_reports:
            summary = self.llm_service.generate_conversational_response(request.query, relevant_reports)
            sources = [report.id for report in relevant_reports]
        else:
            summary = self.llm_service.generate_conversational_response(request.query, [])
            sources = []
        
        return QueryResponse(
            answer=summary,
            sources=sources,
            reports=relevant_reports
        )
    
    def _generate_summary(self, query: str, reports: List[Report]) -> str:
        """Legacy method - now using LLM service exclusively for responses"""
        # This method is kept for backward compatibility but now delegates to LLM service
        return self.llm_service.generate_conversational_response(query, reports)
    
    def get_health_status(self) -> dict:
        """Get service health status"""
        return {
            "status": "healthy" if self.vector_service.index is not None else "not_ready",
            "message": "Service is operational" if self.vector_service.index is not None else "Service not initialized",
            "reports_loaded": len(self.vector_service.reports),
            "vector_index_ready": self.vector_service.index is not None
        }
