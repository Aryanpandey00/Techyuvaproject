"""
Memory Manager: High-level facade orchestrating memory storage, retrieval, and chat turns.
"""
from datetime import datetime, timezone
import logging
import re
import time
from typing import Any, Dict, List, Optional
from core.exceptions import MemoryNotFoundError
from core.schemas import (
    ChatRequest,
    ChatResponse,
    Memory,
    MemoryCreate,
    MemoryRetrieveRequest,
    MemoryRetrieveResponse,
    MemorySearchResult,
    MemoryStatsResponse,
    MemoryStatus,
    MemoryType,
    MemoryUpdate,
    CompressionRequest,
    CompressionResult,
    PruneRequest,
    PruneResult,
)
from database.memory_storage import MemoryStorageService
from embeddings.encoder import BaseEmbeddingEncoder, get_embedding_encoder
from llm.base import BaseLLMProvider
from llm.provider import get_llm_provider
from memory.compressor import MemoryCompressor
from memory.extractor import MemoryExtractor
from memory.forgetter import MemoryForgetter
from memory.retriever import MemoryRetriever
from memory.scorer import MemoryScorer

logger = logging.getLogger(__name__)


class MemoryManager:
    """
    High-level facade orchestrating storage, retrieval, scoring, compression, and conversation loops.
    """

    def __init__(
        self,
        sqlite_db_path: Optional[str] = None,
        chroma_dir: Optional[str] = None,
        embedding_model: Optional[str] = None,
        llm_provider_type: str = "mock",
        openai_api_key: Optional[str] = None,
        openai_model_name: Optional[str] = None,
        openai_base_url: Optional[str] = None,
        in_memory_db: bool = False,
    ) -> None:
        self.encoder: BaseEmbeddingEncoder = get_embedding_encoder(model_name=embedding_model)
        self.storage = MemoryStorageService(
            sqlite_db_path=sqlite_db_path,
            chroma_persist_dir=chroma_dir,
            encoder=self.encoder,
            in_memory=in_memory_db,
        )
        self.sqlite_store = self.storage.sqlite_store
        self.chroma_store = self.storage.chroma_store

        self.llm: BaseLLMProvider = get_llm_provider(
            provider_type=llm_provider_type,
            api_key=openai_api_key,
            model_name=openai_model_name,
            base_url=openai_base_url,
        )
        self.scorer = MemoryScorer()
        self.extractor = MemoryExtractor(llm_provider=self.llm, encoder=self.encoder)
        self.retriever = MemoryRetriever(
            vector_store=self.chroma_store,
            metadata_store=self.sqlite_store,
            encoder=self.encoder,
            scorer=self.scorer,
        )
        self.compressor = MemoryCompressor(
            storage_service=self.storage,
            llm_provider=self.llm,
            encoder=self.encoder,
        )
        self.forgetter = MemoryForgetter(
            vector_store=self.chroma_store,
            metadata_store=self.sqlite_store,
            scorer=self.scorer,
        )
        self._session_turns: Dict[str, List[Dict[str, str]]] = {}

    def create_memory(self, item_create: MemoryCreate) -> Memory:
        """Create and index a new memory."""
        return self.storage.create_memory(item_create)

    def get_memory(self, memory_id: str) -> Optional[Memory]:
        """Fetch memory item by ID."""
        return self.storage.get_memory(memory_id)

    def update_memory(self, memory_id: str, update: MemoryUpdate) -> Memory:
        """Update memory content, metadata, or status."""
        return self.storage.update_memory(memory_id, update)

    def update_status(self, memory_id: str, status: MemoryStatus) -> Memory:
        """Update memory status."""
        return self.storage.update_status(memory_id, status)

    def delete_memory(self, memory_id: str) -> bool:
        """Delete memory from SQLite and ChromaDB."""
        return self.storage.delete_memory(memory_id)

    def list_memories(
        self,
        user_id: Optional[str] = None,
        status: Optional[MemoryStatus] = None,
        memory_type: Optional[MemoryType] = None,
        min_importance: Optional[float] = None,
        min_confidence: Optional[float] = None,
        search_query: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Memory]:
        """List memories with filter parameters."""
        return self.storage.list_memories(
            user_id=user_id,
            status=status,
            memory_type=memory_type,
            min_importance=min_importance,
            min_confidence=min_confidence,
            search_query=search_query,
            limit=limit,
            offset=offset,
        )

    def search_similar(
        self,
        query: str,
        user_id: Optional[str] = None,
        top_k: int = 5,
        min_similarity: float = 0.0,
        memory_type: Optional[MemoryType] = None,
        status: Optional[MemoryStatus] = MemoryStatus.ACTIVE,
    ) -> List[MemorySearchResult]:
        """Execute semantic vector search."""
        return self.storage.search_similar_memories(
            query=query,
            user_id=user_id,
            top_k=top_k,
            min_similarity=min_similarity,
            memory_type=memory_type,
            status=status,
        )

    def retrieve_context(self, request: MemoryRetrieveRequest) -> MemoryRetrieveResponse:
        """Retrieve memories and format prompt context string using MemoryRetriever pipeline."""
        return self.retriever.retrieve(request)

    @staticmethod
    def _should_extract_memory(message: str) -> bool:
        """
        Only store explicit personal statements, not questions or general knowledge queries.
        Rejects questions, query commands, conversational chatter, and non-personal topics.
        """
        text = (message or "").strip()
        if not text:
            return False
        if "?" in text:
            return False
        lowered = text.lower()
        # Reject question openings or command verbs
        question_starters = (
            "what", "why", "how", "when", "where", "who", "which",
            "can", "could", "should", "would", "do", "does", "did",
            "is", "are", "am", "was", "were", "tell me", "explain",
            "describe", "show me", "define", "list", "summarize",
            "search", "give me", "please", "help"
        )
        if any(lowered.startswith(qs + " ") or lowered == qs for qs in question_starters):
            return False
        if re.search(r"\b(?:what is|who is|where is|when was|why is|how do|how does|capital of|national animal)\b", lowered):
            return False

        # Only extract if there is explicit personal intent or declarative self-description
        personal_markers = (
            "i am ", "i'm ", "my name ", "my favorite ", "my pref",
            "i prefer ", "i like ", "i love ", "i work ", "i live ",
            "i study ", "i'm studying ", "i am studying ", "i am building ",
            "i'm building ", "i am developing ", "my goal ", "my job ",
            "my project ", "my university ", "my college ", "call me ",
            "remember that i ", "remember i ", "i have a "
        )
        return any(pm in lowered for pm in personal_markers)

    def process_chat(self, chat_req: ChatRequest) -> ChatResponse:
        """
        Process chat turn through the Resource-Efficient Long-Term Memory Layer:
        1. Retrieve candidate nearest neighbors from ChromaDB and SQLite
        2. Relevance Gate: filter out weak/irrelevant memories
        3. Context Budget: enforce compact limits (max items, max tokens, max characters)
        4. Context Policy: inject compact relevant context or send query-only
        5. Natural LLM Generation (Mock or OpenAI)
        6. Compute real-time resource-efficiency metrics (tokens saved, % reduction, latency)
        """
        t_start = time.perf_counter()

        # Resolve compact conversational history for follow-up disambiguation
        recent_turns = chat_req.recent_history if chat_req.recent_history is not None else self._session_turns.get(chat_req.user_id, [])

        retrieval_query = chat_req.message
        if recent_turns and len(chat_req.message.split()) <= 12:
            lowered_msg = chat_req.message.lower()
            followup_indicators = [
                "it", "this", "that", "these", "those",
                "ye", "yeh", "wo", "woh", "iska", "iski", "iske", "isko",
                "uske", "uski", "usko", "inhe", "unhe", "inka", "unka",
                "isme", "usme", "kisme", "kis yug", "kaun se yug",
                "why is it", "how does it", "example", "explain simply",
            ]
            has_followup_ref = any(
                re.search(rf"\b{re.escape(ind)}\b", lowered_msg)
                for ind in followup_indicators
            )
            if has_followup_ref:
                last_user_msg = next((t["content"] for t in reversed(recent_turns) if t.get("role") == "user"), "")
                if last_user_msg:
                    retrieval_query = f"{last_user_msg} {chat_req.message}"

        # 1. Resolve relevance cutoff
        rel_cutoff = (
            chat_req.relevance_threshold
            if chat_req.relevance_threshold is not None
            else getattr(self.retriever, "relevance_threshold", 0.65)
        )

        # Retrieve candidates with resolved relevance cutoff
        retrieve_resp = self.retrieve_context(MemoryRetrieveRequest(
            query=retrieval_query,
            user_id=chat_req.user_id,
            top_k=chat_req.top_k_memories,
            relevance_threshold=rel_cutoff,
        ))

        # 2. Relevance Gate filtering: final composite score is the single authoritative cutoff.
        relevant_candidates = [
            r for r in retrieve_resp.results
            if r.final_score >= rel_cutoff
        ]

        # 3. Context Budget Enforcement
        max_injected = 3
        max_tokens = chat_req.max_context_tokens or 250
        max_chars = max_tokens * 4

        injected_memories: List[MemorySearchResult] = []
        cumulative_chars = 0

        for candidate in relevant_candidates:
            if len(injected_memories) >= max_injected:
                break
            cand_len = len(candidate.memory.content)
            if cumulative_chars + cand_len > max_chars:
                break
            injected_memories.append(candidate)
            cumulative_chars += cand_len

        # 4. Context Policy: Compact context or Query-only
        if injected_memories:
            context_string = self.retriever.format_context_string(injected_memories)
            memory_used = True
            relevance_status = "memory_injected"
        else:
            context_string = None
            memory_used = False
            relevance_status = "memory_filtered" if retrieve_resp.results else "general_knowledge"

        # 5. Natural LLM response generation with compact conversation history
        reply = self.llm.generate_response(
            prompt=chat_req.message,
            context=context_string,
            conversation_history=recent_turns,
        )

        # Update compact session turns for follow-up disambiguation (capped at last 4 messages)
        if chat_req.user_id not in self._session_turns:
            self._session_turns[chat_req.user_id] = []
        self._session_turns[chat_req.user_id].append({"role": "user", "content": chat_req.message})
        self._session_turns[chat_req.user_id].append({"role": "assistant", "content": reply})
        if len(self._session_turns[chat_req.user_id]) > 4:
            self._session_turns[chat_req.user_id] = self._session_turns[chat_req.user_id][-4:]

        if reply:
            reply = re.sub(r"\[RELEVANT STORED MEMORIES\]", "", reply, flags=re.IGNORECASE)
            reply = re.sub(r"\[MEMORY CONTEXT\]", "", reply, flags=re.IGNORECASE)
            reply = re.sub(r"^Based on your stored memory context:\s*", "", reply, flags=re.IGNORECASE).strip()

        # 6. Memory Extraction for long-term facts
        new_memories: List[Memory] = []
        if chat_req.auto_extract_memories and self._should_extract_memory(chat_req.message):
            extracted_creates = self.extractor.extract_from_turn(
                user_message=chat_req.message,
                assistant_reply=reply
            )
            for m_create in extracted_creates:
                m_create.user_id = chat_req.user_id
                created = self.create_memory(m_create)
                new_memories.append(created)

        # 7. Compute real-time Resource Efficiency Metrics
        retrieval_latency_ms = round((time.perf_counter() - t_start) * 1000.0, 2)
        injected_tokens = (len(context_string) // 4) if context_string else 0

        # Estimate uncompressed baseline context (all active memories for user formatted into prompt if sent naively)
        try:
            active_mems = self.list_memories(user_id=chat_req.user_id, status=MemoryStatus.ACTIVE)
            if active_mems:
                lines = ["[RELEVANT STORED MEMORIES]"]
                for idx, m in enumerate(active_mems, start=1):
                    m_type = m.memory_type.value if hasattr(m.memory_type, "value") else str(m.memory_type)
                    lines.append(f"{idx}. [{m_type.upper()}] {m.content}")
                baseline_str = "\n".join(lines)
                uncompressed_tokens = max(len(baseline_str) // 4, injected_tokens)
            else:
                uncompressed_tokens = injected_tokens
        except Exception:
            uncompressed_tokens = injected_tokens

        if memory_used and uncompressed_tokens > 0:
            tokens_saved = max(0, uncompressed_tokens - injected_tokens)
            pct_reduction = round((tokens_saved / float(uncompressed_tokens)) * 100.0, 1)
        else:
            # Query answered without memory: zero memory context tokens used
            tokens_saved = 0
            pct_reduction = 0.0

        return ChatResponse(
            reply=reply,
            retrieved_memories=retrieve_resp.results,
            injected_memories=injected_memories,
            new_extracted_memories=new_memories,
            context_tokens_used=injected_tokens,
            context_tokens_saved=tokens_saved,
            context_reduction_percent=pct_reduction,
            retrieval_latency_ms=retrieval_latency_ms,
            memory_used=memory_used,
            relevance_status=relevance_status,
        )

    def get_stats(self, user_id: Optional[str] = None) -> MemoryStatsResponse:
        """Aggregate statistics."""
        return self.storage.get_stats(user_id=user_id)

    def compress_memories(self, req: Optional[CompressionRequest] = None) -> CompressionResult:
        """Trigger clustering and memory compression."""
        request = req or CompressionRequest()
        return self.compressor.compress_memories(
            memory_ids=request.memory_ids,
            user_id=request.user_id,
            similarity_threshold=request.similarity_threshold,
            min_cluster_size=request.min_cluster_size,
            max_memory_length=request.max_memory_length,
            dry_run=request.dry_run,
        )

    def prune_memories(self, req: Optional[PruneRequest] = None) -> PruneResult:
        """Execute decay-based and inactivity-based pruning of obsolete memories."""
        request = req or PruneRequest()
        return self.forgetter.prune_memories(request)
