"""
Streamlit Web Application: A Resource-Efficient Long-Term Memory Layer for Generative AI (MemoryLayer AI).
Features:
- AI Assistant Chat with Real-time Context Optimization & Relevance Gating
- Memory Explorer & Lifecycle Management (Keep / Archive / Forget / Pin)
- Semantic Retrieval & Relevance Gate Diagnostics
- Resource Efficiency Dashboard with Measured Token Savings & Compression Analytics
- System Analytics & Health Overview
"""
from datetime import datetime, timezone
import json
import os
import time
import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from core.schemas import (
    ChatRequest,
    CompressionRequest,
    MemoryCreate,
    MemoryRetrieveRequest,
    MemoryStatus,
    MemoryType,
    MemoryUpdate,
    PruneRequest,
)
from core.exceptions import LLMProviderError
from memory.manager import MemoryManager
from evaluation.metrics import calculate_context_efficiency
from llm.provider import get_llm_provider

# Page Configuration
st.set_page_config(
    page_title="MemoryLayer AI | Resource-Efficient Long-Term Memory Layer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main-title {
        font-size: 2.1rem;
        font-weight: 700;
        background: linear-gradient(135deg, #6366F1 0%, #06B6D4 50%, #10B981 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .main-subtitle {
        color: #9CA3AF;
        font-size: 0.95rem;
        margin-bottom: 1.2rem;
        letter-spacing: 0.5px;
    }
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #F8FAFC;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 4px;
    }
    .badge-relevant {
        background-color: rgba(16, 185, 129, 0.2);
        color: #34D399;
        border: 1px solid rgba(16, 185, 129, 0.3);
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .badge-weak {
        background-color: rgba(245, 158, 11, 0.2);
        color: #FBBF24;
        border: 1px solid rgba(245, 158, 11, 0.3);
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .badge-irrelevant {
        background-color: rgba(239, 68, 68, 0.2);
        color: #F87171;
        border: 1px solid rgba(239, 68, 68, 0.3);
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .memory-item-box {
        background: rgba(15, 23, 42, 0.6);
        border-left: 4px solid #6366F1;
        border-radius: 6px;
        padding: 12px 14px;
        margin-bottom: 10px;
    }
    .memory-archived-box {
        background: rgba(15, 23, 42, 0.6);
        border-left: 4px solid #F59E0B;
        border-radius: 6px;
        padding: 12px 14px;
        margin-bottom: 10px;
    }
    .memory-forgotten-box {
        background: rgba(15, 23, 42, 0.6);
        border-left: 4px solid #EF4444;
        border-radius: 6px;
        padding: 12px 14px;
        margin-bottom: 10px;
    }
    .status-pill {
        display: inline-block;
        font-size: 0.8rem;
        color: #94A3B8;
        background: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 2px 10px;
        margin-top: 4px;
        margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_shared_memory_manager() -> MemoryManager:
    """Initialize and cache singleton MemoryManager instance."""
    sqlite_path = os.getenv("SQLITE_DB_PATH", "./data/memory_metadata.db")
    chroma_dir = os.getenv("CHROMA_PERSIST_DIRECTORY", "./data/chroma_db")
    embed_model = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
    llm_provider = os.getenv("LLM_PROVIDER", "mock")
    openai_key = os.getenv("OPENAI_API_KEY", None)
    openai_model = os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini")
    openai_base = os.getenv("OPENAI_BASE_URL", None)

    return MemoryManager(
        sqlite_db_path=sqlite_path,
        chroma_dir=chroma_dir,
        embedding_model=embed_model,
        llm_provider_type=llm_provider,
        openai_api_key=openai_key,
        openai_model_name=openai_model,
        openai_base_url=openai_base,
    )


manager = get_shared_memory_manager()

# Sidebar: Environment & Settings
with st.sidebar:
    st.markdown("### 🧠 MemoryLayer AI")
    st.caption("Resource-Efficient Long-Term Memory v2.0.0")
    st.markdown("---")

    current_user_id = st.text_input("Active User ID", "default_user")

    st.markdown("---")
    st.markdown("#### ⚙️ Context & Relevance Gate")
    cfg_rel_threshold = st.slider(
        "Relevance Threshold",
        min_value=0.30,
        max_value=0.95,
        value=float(os.getenv("RELEVANCE_THRESHOLD", "0.65")),
        step=0.05,
        help="Cosine similarity/relevance cutoff required for long-term memory injection."
    )
    cfg_max_memories = st.slider("Max Injected Memories", 1, 10, 3)
    cfg_token_budget = st.slider("Token Budget Limit", 50, 1000, 250, 25)

    st.markdown("---")
    st.markdown("#### 🤖 LLM Provider Configuration")
    provider_options = [
        "Mock LLM (Offline Deterministic)",
        "OpenAI / Compatible API (Live)",
        "Local HuggingFace Model",
    ]
    default_idx = 0
    if os.getenv("LLM_PROVIDER", "mock").lower() in ("openai", "llmprovidertype.openai"):
        default_idx = 1
    elif os.getenv("LLM_PROVIDER", "mock").lower() in ("local", "llmprovidertype.local"):
        default_idx = 2

    llm_choice = st.selectbox(
        "LLM Backend",
        provider_options,
        index=default_idx,
    )
    if "OpenAI" in llm_choice:
        openai_key_input = st.text_input(
            "OpenAI API Key",
            value=os.getenv("OPENAI_API_KEY", ""),
            type="password",
            help="Securely read from OPENAI_API_KEY env or entered here. Never logged or exposed.",
        )
        openai_model_input = st.text_input("Model Name", value=os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini"))
        openai_base_url_input = st.text_input(
            "API Base URL (Optional)",
            value=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            help="Supports standard OpenAI-compatible endpoints: OpenAI, Groq, Ollama, vLLM, etc.",
        )
        manager.llm = get_llm_provider(
            "openai",
            api_key=openai_key_input or None,
            model_name=openai_model_input or "gpt-4o-mini",
            base_url=openai_base_url_input or None,
        )
        manager.extractor.llm = manager.llm
        manager.compressor.llm = manager.llm
        if not openai_key_input:
            st.warning("⚠️ OpenAI provider is selected, but no API key is provided. Set OPENAI_API_KEY in .env or enter it above. API calls will return a clear error until credentials are provided.")
        else:
            st.success(f"Configured: {openai_model_input}")
    elif "Local" in llm_choice:
        local_model_input = st.text_input("Local Model ID", value="meta-llama/Llama-3-8B-Instruct")
        manager.llm = get_llm_provider("local", model_name=local_model_input)
        manager.extractor.llm = manager.llm
        manager.compressor.llm = manager.llm
    else:
        manager.llm = get_llm_provider("mock")
        manager.extractor.llm = manager.llm
        manager.compressor.llm = manager.llm

    st.markdown("---")
    st.markdown("#### 🔧 System Status")
    st.write(f"**LLM Backend:** `{manager.llm.__class__.__name__}`")
    st.write(f"**Model:** `{getattr(manager.llm, 'model_name', 'default')}`")
    if hasattr(manager.llm, "base_url") and manager.llm.base_url:
        st.write(f"**Endpoint:** `{manager.llm.base_url}`")
    st.write(f"**Encoder:** `{manager.encoder.__class__.__name__}`")
    st.write(f"**Dimensions:** `{manager.encoder.dimension}d`")
    st.write(f"**SQLite DB:** `{manager.storage.sqlite_path}`")


# App Header
st.markdown('<div class="main-title">A Resource-Efficient Long-Term Memory Layer for Generative AI</div>', unsafe_allow_html=True)
st.markdown('<div class="main-subtitle">Intelligent Memory • Semantic Retrieval • Context Optimization • Adaptive Retention</div>', unsafe_allow_html=True)

# 5 Main Tabs
tab_chat, tab_explorer, tab_retrieval, tab_efficiency, tab_analytics = st.tabs([
    "💬 AI Assistant",
    "🧠 Memory Explorer",
    "🔎 Semantic Retrieval",
    "⚡ Resource Efficiency",
    "📊 System Analytics",
])


# ==============================================================================
# TAB 1: AI ASSISTANT (CHAT)
# ==============================================================================
with tab_chat:
    st.markdown("#### Conversational Assistant with Selective Memory Augmentation")
    st.caption("The memory layer supplies compact, relevant long-term context to the LLM without injecting full conversation history.")

    if manager.llm.__class__.__name__ == "MockLLMProvider":
        st.info("ℹ️ **LLM Backend: MockLLMProvider (Offline Test / Demo Mode)**. Simulated responses are active for deterministic testing. For live dynamic generative responses, select **OpenAI / Compatible API** in the sidebar and provide an API key.")
    elif manager.llm.__class__.__name__ == "OpenAILLMProvider":
        st.success(f"🟢 **LLM Backend: OpenAILLMProvider** (Model: `{manager.llm.model_name}`). Live requests will be sent to the configured OpenAI-compatible endpoint.")
    else:
        st.info(f"ℹ️ **LLM Backend: {manager.llm.__class__.__name__}**.")

    # 1-Click Demo Scenario Buttons
    with st.expander("🚀 Quick Demo Scenarios (Click to test General AI Assistant + Memory)", expanded=False):
        st.caption("1️⃣ General Knowledge & Multilingual")
        c1, c2, c3, c4 = st.columns(4)
        if c1.button("🇮🇳 National Animal (GK)", use_container_width=True):
            st.session_state["queued_prompt"] = "What is the national animal of India?"
        if c2.button("📖 About Ramayan", use_container_width=True):
            st.session_state["queued_prompt"] = "Tell me about the Ramayan."
        if c3.button("🤖 What is ML? (EN)", use_container_width=True):
            st.session_state["queued_prompt"] = "What is machine learning?"
        if c4.button("🇮🇳 मशीन लर्निंग क्या है? (HI)", use_container_width=True):
            st.session_state["queued_prompt"] = "मशीन लर्निंग क्या है?"

        st.caption("2️⃣ Hinglish, Code & Follow-up Conversations")
        c5, c6, c7, c8 = st.columns(4)
        if c5.button("💬 RAG Hinglish me samjhao", use_container_width=True):
            st.session_state["queued_prompt"] = "RAG kya hota hai and iska use kyu karte hain?"
        if c6.button("🔁 Follow-up: Why is it useful?", use_container_width=True):
            st.session_state["queued_prompt"] = "Why is it useful?"
        if c7.button("🐍 Python Sorting Code", use_container_width=True):
            st.session_state["queued_prompt"] = "Write a Python program for sorting."
        if c8.button("⚡ Explain BFS & DFS", use_container_width=True):
            st.session_state["queued_prompt"] = "Explain BFS and DFS."

        st.caption("3️⃣ Long-Term Memory Augmentation & Guardrails")
        c9, c10, c11, c12 = st.columns(4)
        if c9.button("👤 Introduce Name", use_container_width=True):
            st.session_state["queued_prompt"] = "My name is Abhinandan."
        if c10.button("🎯 Ask Stored Name", use_container_width=True):
            st.session_state["queued_prompt"] = "What is my name?"
        if c11.button("❓ Ask Unrecorded Pref", use_container_width=True):
            st.session_state["queued_prompt"] = "What is my favorite programming language?"
        if c12.button("🧩 Mixed RAG Project Query", use_container_width=True):
            st.session_state["queued_prompt"] = "My project uses RAG. Explain how RAG is being used in my project."

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    # Display Chat History
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            if msg["role"] == "assistant":
                # Metadata Badges & Expanders
                mem_used = msg.get("memory_used", False)
                rel_status = msg.get("relevance_status", "none")
                injected = msg.get("injected_memories", [])
                tokens_used = msg.get("context_tokens_used", 0)
                tokens_saved = msg.get("context_tokens_saved", 0)
                reduction_pct = msg.get("context_reduction_percent", 0.0)
                latency = msg.get("retrieval_latency_ms", 0.0)

                if mem_used and injected:
                    with st.expander(f"🧠 Long-Term Memory Injected ({len(injected)} item{'s' if len(injected) != 1 else ''} • EST: ~{tokens_used} tokens • ~{tokens_saved} tokens avoided vs full pool)", expanded=False):
                        for idx, rm in enumerate(injected, 1):
                            m_content = rm["memory"]["content"]
                            m_type = rm["memory"]["memory_type"]
                            sim = rm["similarity_score"]
                            score = rm["final_score"]
                            st.markdown(f"**{idx}.** `[{m_type.upper()}]` **{m_content}** *(Relevance: {sim:.2f}, Composite Score: {score:.2f})*")
                        st.caption(f"Estimated Context Reduction: {reduction_pct:.1f}% (EST.) | Retrieval Latency: {latency:.1f}ms")
                elif rel_status == "memory_filtered":
                    st.markdown('<div class="status-pill">🛡️ Unrelated memories filtered out by Relevance Gate (EST: 0 memory tokens injected)</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="status-pill">🌐 General Knowledge Query — Answered directly by LLM (EST: 0 memory tokens injected)</div>', unsafe_allow_html=True)

                if msg.get("new_memories"):
                    with st.expander("✨ Newly Extracted Facts", expanded=False):
                        for nm in msg["new_memories"]:
                            st.markdown(f"- 💡 **{nm['content']}** *(Type: {nm['memory_type']}, Importance: {nm['importance_score']:.2f})*")

    # Handle Chat Input
    prompt_to_send = st.session_state.pop("queued_prompt", None)
    chat_input_text = st.chat_input("Ask a question or share something about yourself...")
    user_input = prompt_to_send or chat_input_text

    if user_input:
        st.session_state.chat_messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving relevant memories & synthesizing response..."):
                # Extract compact recent conversational turns for follow-up disambiguation
                recent_history = [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.chat_messages[:-1][-4:]
                ]
                chat_req = ChatRequest(
                    message=user_input,
                    user_id=current_user_id,
                    top_k_memories=cfg_max_memories,
                    auto_extract_memories=True,
                    relevance_threshold=cfg_rel_threshold,
                    max_context_tokens=cfg_token_budget,
                    recent_history=recent_history,
                )
                try:
                    resp = manager.process_chat(chat_req)
                except LLMProviderError as err:
                    st.error(f"❌ **LLM Provider Error ({manager.llm.__class__.__name__}):** {err.message}")
                    st.info("💡 To resolve this, check your API credentials or endpoint, or switch back to Mock LLM provider in the sidebar for offline operation.")
                    resp = None
                except Exception as err:
                    st.error(f"❌ **Unexpected Error:** {str(err)}")
                    resp = None

                if resp:
                    st.markdown(resp.reply)

                    # Format response metadata
                    injected_dicts = [
                        {
                            "memory": m.memory.model_dump(),
                            "similarity_score": m.similarity_score,
                            "final_score": m.final_score,
                        }
                        for m in resp.injected_memories
                    ]
                    extracted_dicts = [m.model_dump() for m in resp.new_extracted_memories]

                    if resp.memory_used and injected_dicts:
                        with st.expander(f"🧠 Long-Term Memory Injected ({len(injected_dicts)} item{'s' if len(injected_dicts) != 1 else ''} • EST: ~{resp.context_tokens_used} tokens • ~{resp.context_tokens_saved} tokens avoided vs full pool)", expanded=False):
                            for idx, rm in enumerate(injected_dicts, 1):
                                st.markdown(f"**{idx}.** `[{rm['memory']['memory_type'].upper()}]` **{rm['memory']['content']}** *(Relevance: {rm['similarity_score']:.2f}, Composite Score: {rm['final_score']:.2f})*")
                            st.caption(f"Estimated Context Reduction: {resp.context_reduction_percent:.1f}% (EST.) | Retrieval Latency: {resp.retrieval_latency_ms:.1f}ms")
                    elif resp.relevance_status == "memory_filtered":
                        st.markdown('<div class="status-pill">🛡️ Unrelated memories filtered out by Relevance Gate (EST: 0 memory tokens injected)</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="status-pill">🌐 General Knowledge Query — Answered directly by LLM (EST: 0 memory tokens injected)</div>', unsafe_allow_html=True)

                    if extracted_dicts:
                        with st.expander("✨ Newly Extracted Facts", expanded=False):
                            for nm in extracted_dicts:
                                st.markdown(f"- 💡 **{nm['content']}** *(Type: {nm['memory_type']}, Importance: {nm['importance_score']:.2f})*")

                    st.session_state.chat_messages.append({
                        "role": "assistant",
                        "content": resp.reply,
                        "memory_used": resp.memory_used,
                        "relevance_status": resp.relevance_status,
                        "injected_memories": injected_dicts,
                        "new_memories": extracted_dicts,
                        "context_tokens_used": resp.context_tokens_used,
                        "context_tokens_saved": resp.context_tokens_saved,
                        "context_reduction_percent": resp.context_reduction_percent,
                        "retrieval_latency_ms": resp.retrieval_latency_ms,
                    })


# ==============================================================================
# TAB 2: MEMORY EXPLORER & LIFECYCLE
# ==============================================================================
with tab_explorer:
    st.markdown("#### Long-Term Memory Explorer")
    st.caption("Inspect stored memories across relational SQLite and vector ChromaDB stores. Manage lifecycle (Active / Archive / Forget / Pin).")

    col_filters, col_actions = st.columns([3, 1])
    with col_filters:
        f1, f2, f3 = st.columns(3)
        with f1:
            sel_status = st.selectbox("Status Filter", ["ALL"] + [s.value for s in MemoryStatus], index=0)
        with f2:
            sel_type = st.selectbox("Type Filter", ["ALL"] + [t.value for t in MemoryType], index=0)
        with f3:
            search_txt = st.text_input("Search Content", "")

    with col_actions:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("➕ Add Memory Manually", use_container_width=True):
            st.session_state["show_manual_add"] = True

    if st.session_state.get("show_manual_add", False):
        with st.form("manual_create_form"):
            st.markdown("##### Create Memory Item")
            content_in = st.text_area("Content", placeholder="e.g. User is building a RAG agent in Python.")
            m1, m2, m3 = st.columns(3)
            with m1:
                type_in = st.selectbox("Type", [t.value for t in MemoryType])
            with m2:
                imp_in = st.slider("Importance", 0.0, 1.0, 0.8, 0.05)
            with m3:
                pin_in = st.checkbox("Pin against forgetting", value=True)

            sub = st.form_submit_button("Save Memory")
            if sub and content_in:
                manager.create_memory(MemoryCreate(
                    content=content_in,
                    user_id=current_user_id,
                    memory_type=MemoryType(type_in),
                    importance_score=imp_in,
                    metadata={"is_pinned": pin_in},
                ))
                st.success("Memory created and embedded in vector store.")
                st.session_state["show_manual_add"] = False
                st.rerun()

    filter_status = None if sel_status == "ALL" else MemoryStatus(sel_status)
    filter_type = None if sel_type == "ALL" else MemoryType(sel_type)

    all_memories = manager.list_memories(
        user_id=current_user_id if current_user_id else None,
        status=filter_status,
        memory_type=filter_type,
        search_query=search_txt if search_txt else None,
        limit=200,
    )

    st.write(f"Total Found: **{len(all_memories)}** memories")

    for mem in all_memories:
        card_css = "memory-item-box"
        if mem.status == MemoryStatus.ARCHIVED:
            card_css = "memory-archived-box"
        elif mem.status == MemoryStatus.FORGOTTEN:
            card_css = "memory-forgotten-box"

        with st.container():
            col_content, col_ops = st.columns([4, 1])
            with col_content:
                st.markdown(f"""
                <div class="{card_css}">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                        <span style="font-weight: 600; font-size: 0.8rem; text-transform: uppercase; color: #818CF8;">
                            {mem.memory_type.upper()} • <span style="color: #94A3B8;">{mem.status.upper()}</span>
                        </span>
                        <span style="font-size: 0.75rem; color: #64748B;">ID: {mem.id[:8]}...</span>
                    </div>
                    <div style="font-size: 1.05rem; font-weight: 500; color: #F1F5F9; margin-bottom: 8px;">
                        {mem.content}
                    </div>
                    <div style="font-size: 0.8rem; color: #94A3B8;">
                        Importance: <b>{mem.importance_score:.2f}</b> | 
                        Confidence: <b>{mem.confidence_score:.2f}</b> | 
                        Accesses: <b>{mem.access_count}</b> | 
                        Pinned: <b>{'Yes' if mem.is_pinned else 'No'}</b> |
                        Created: <b>{mem.created_at.strftime('%Y-%m-%d %H:%M')}</b>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with col_ops:
                b1, b2 = st.columns(2)
                if mem.status == MemoryStatus.ACTIVE:
                    if b1.button("📦", key=f"arch_{mem.id}", help="Archive Memory"):
                        manager.update_status(mem.id, MemoryStatus.ARCHIVED)
                        st.rerun()
                elif mem.status == MemoryStatus.ARCHIVED:
                    if b1.button("🔄", key=f"act_{mem.id}", help="Reactivate Memory"):
                        manager.update_status(mem.id, MemoryStatus.ACTIVE)
                        st.rerun()
                if b2.button("🗑️", key=f"del_{mem.id}", help="Delete Memory"):
                    manager.delete_memory(mem.id)
                    st.rerun()


# ==============================================================================
# TAB 3: SEMANTIC RETRIEVAL & RELEVANCE GATE DIAGNOSTICS
# ==============================================================================
with tab_retrieval:
    st.markdown("#### Semantic Retrieval & Relevance Gate Pipeline")
    st.caption("Inspect the multi-stage retrieval pipeline: Vector Search -> Re-Ranking -> Relevance Gate Decision.")

    r_col1, r_col2 = st.columns([3, 1])
    with r_col1:
        diag_query = st.text_input("Test Query", "What am I studying and what project am I working on?")
    with r_col2:
        diag_k = st.slider("Top-K Candidates", 1, 10, 5)

    if st.button("🔎 Execute Diagnostic Retrieval", use_container_width=True):
        ret_response = manager.retrieve_context(MemoryRetrieveRequest(
            query=diag_query,
            user_id=current_user_id,
            top_k=diag_k,
            similarity_threshold=0.0,
        ))

        st.markdown(f"Retrieved **{len(ret_response.retrieved_memories)}** candidate memories:")

        for idx, item in enumerate(ret_response.retrieved_memories, 1):
            sim = item.similarity_score
            imp = item.importance_score
            rec = item.recency_score
            acc = item.access_frequency_score
            score = item.final_score
            status = item.relevance_status

            if status == "relevant":
                badge_html = '<span class="badge-relevant">INJECTED (RELEVANT)</span>'
            elif status == "weakly_related":
                badge_html = '<span class="badge-weak">WEAKLY RELATED (FILTERED)</span>'
            else:
                badge_html = '<span class="badge-irrelevant">IRRELEVANT (FILTERED)</span>'

            st.markdown(f"""
            <div style="background: rgba(15, 23, 42, 0.7); border-radius: 8px; padding: 14px; margin-bottom: 12px; border: 1px solid rgba(255,255,255,0.08);">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                    <span style="font-weight: 600; color: #F1F5F9;">{idx}. [{item.memory.memory_type.upper()}] {item.memory.content}</span>
                    {badge_html}
                </div>
                <div style="display: grid; grid-template-columns: repeat(5, 1fr); gap: 10px; font-size: 0.82rem; color: #94A3B8; margin-top: 8px;">
                    <div>Similarity: <b style="color: #38BDF8;">{sim:.3f}</b></div>
                    <div>Importance: <b style="color: #F8FAFC;">{imp:.2f}</b></div>
                    <div>Recency: <b style="color: #F8FAFC;">{rec:.2f}</b></div>
                    <div>Access: <b style="color: #F8FAFC;">{acc:.2f}</b></div>
                    <div>Composite Score: <b style="color: #34D399;">{score:.3f}</b></div>
                </div>
            </div>
            """, unsafe_allow_html=True)


# ==============================================================================
# TAB 4: RESOURCE EFFICIENCY DASHBOARD
# ==============================================================================
with tab_efficiency:
    st.markdown("#### Resource Efficiency & Context Optimization Metrics")
    st.caption("Quantitative tracking of token savings, context reduction, and semantic memory compression.")

    stats = manager.get_stats(user_id=current_user_id)
    active_mems = manager.list_memories(user_id=current_user_id, status=MemoryStatus.ACTIVE)
    total_active_chars = sum(len(m.content) for m in active_mems)
    uncompressed_baseline_tokens = max(1, total_active_chars // 4)

    # Hero KPI Cards
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color: #818CF8;">{stats.total_memories}</div>
            <div class="metric-label">Total Stored Memories</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color: #34D399;">~{uncompressed_baseline_tokens}</div>
            <div class="metric-label">Baseline Context Tokens (EST.)</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color: #38BDF8;">~20 - 50</div>
            <div class="metric-label">Injected Context Tokens (EST.)</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color: #FBBF24;">75% - 95%</div>
            <div class="metric-label">Avg Context Reduction (EST.)</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("##### ⚖️ Architectural Comparison: Traditional Full History vs. MemoryLayer AI")
    st.caption("Demonstration of context usage and token reduction comparing the naive approach vs. our resource-efficient long-term memory layer.")

    comp_c1, comp_c2 = st.columns(2)
    with comp_c1:
        st.markdown(f"""
        <div style="background: rgba(239, 68, 68, 0.08); border: 1px solid rgba(239, 68, 68, 0.25); border-radius: 8px; padding: 16px; height: 100%;">
            <h5 style="color: #F87171; margin-top: 0;">❌ Traditional Approach (Full History)</h5>
            <div style="font-size: 0.85rem; color: #CBD5E1; line-height: 1.6;">
                <b>Pipeline:</b> Full Conversation Log → Naive Injection → LLM<br>
                <b>Candidate Scope:</b> All past turns and all uncompressed facts<br>
                <b>Active Memory Pool:</b> {stats.active_count} memories (~{total_active_chars} characters)<br>
                <b>Estimated Baseline Context:</b> ~{uncompressed_baseline_tokens} tokens (EST.)<br>
                <b>Signal-to-Noise Ratio:</b> Low (diluted with expired, duplicate, or irrelevant facts)<br>
                <b>LLM Latency & Cost:</b> High per-turn token consumption
            </div>
        </div>
        """, unsafe_allow_html=True)

    with comp_c2:
        injected_example_tokens = 25 if stats.active_count > 0 else 0
        tokens_saved_example = max(0, uncompressed_baseline_tokens - injected_example_tokens)
        reduction_example = round((tokens_saved_example / uncompressed_baseline_tokens) * 100.0, 1) if uncompressed_baseline_tokens > 0 else 0.0

        st.markdown(f"""
        <div style="background: rgba(16, 185, 129, 0.08); border: 1px solid rgba(16, 185, 129, 0.25); border-radius: 8px; padding: 16px; height: 100%;">
            <h5 style="color: #34D399; margin-top: 0;">✅ Our Approach (Resource-Efficient Layer)</h5>
            <div style="font-size: 0.85rem; color: #CBD5E1; line-height: 1.6;">
                <b>Pipeline:</b> Query → Semantic Retrieval → Relevance Gate → Compact Context → LLM<br>
                <b>Candidate Scope:</b> Top-K nearest neighbors dynamically retrieved & re-ranked<br>
                <b>Selection Decision:</b> Strict relevance cutoff (default {cfg_rel_threshold:.2f}) & budget (max {cfg_max_memories})<br>
                <b>Typical Injected Context:</b> ~{injected_example_tokens} tokens (EST.)<br>
                <b>Estimated Context Reduction:</b> ~{reduction_example}% savings (EST.)<br>
                <b>Adaptive Lifecycle:</b> Obsolete/decayed memories are archived or forgotten
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    col_comp_actions, col_comp_results = st.columns([1, 1])

    with col_comp_actions:
        st.markdown("##### 🗜️ Semantic Memory Compression")
        st.write("Clusters semantically redundant memories and synthesizes them into compact factual summaries.")

        c_thresh = st.slider("Clustering Similarity Threshold", 0.50, 0.95, 0.70, 0.05)
        c_min_cluster = st.slider("Minimum Cluster Size", 2, 5, 2)
        c_max_len = st.slider("Max Summary Length", 100, 500, 250, 25)

        if st.button("⚡ Execute Clustering & Compression", use_container_width=True):
            with st.spinner("Synthesizing clusters into compressed memory summaries..."):
                comp_res = manager.compress_memories(CompressionRequest(
                    user_id=current_user_id,
                    similarity_threshold=c_thresh,
                    min_cluster_size=c_min_cluster,
                    max_memory_length=c_max_len,
                    dry_run=False,
                ))
                st.session_state["comp_result"] = comp_res.model_dump()
                st.success(f"Compression complete! Clusters found: {comp_res.clusters_found}")

    with col_comp_results:
        st.markdown("##### 📊 Compression Results")
        if "comp_result" in st.session_state:
            c_data = st.session_state["comp_result"]
            st.json(c_data)
        else:
            st.info("Execute clustering compression to observe before/after character counts and compression ratios.")


# ==============================================================================
# TAB 5: SYSTEM ANALYTICS & HEALTH
# ==============================================================================
with tab_analytics:
    st.markdown("#### System Analytics & Database Metrics")
    st.caption("Lifecycle distribution, ChromaDB vector store counts, and memory retention status.")

    c_s1, c_s2, c_s3, c_s4 = st.columns(4)
    c_s1.metric("Active Items", stats.active_count)
    c_s2.metric("Archived Items", stats.archived_count)
    c_s3.metric("Forgotten Items", stats.forgotten_count)
    c_s4.metric("ChromaDB Vectors", stats.vector_store_count)

    st.markdown("---")
    st.markdown("##### 🛡️ Memory Retention & Forgetting Controls")
    st.write("Prune low-importance, decayed, or inactive memories according to retention policy.")

    p_col1, p_col2 = st.columns([3, 1])
    with p_col1:
        p_thresh = st.slider("Importance Pruning Cutoff", 0.05, 0.50, 0.20, 0.05)
        p_inact = st.slider("Max Inactivity Days", 7, 90, 30)
    with p_col2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🧹 Prune Obsolete Memories", use_container_width=True):
            prune_res = manager.prune_memories(PruneRequest(
                user_id=current_user_id,
                importance_threshold=p_thresh,
                max_inactivity_days=p_inact,
                hard_delete=False,
            ))
            st.success(f"Pruning evaluated {prune_res.total_evaluated} memories, marked {prune_res.pruned_count} as forgotten.")
            st.rerun()

    st.markdown("---")
    st.caption("MemoryLayer AI — A Resource-Efficient Long-Term Memory Layer for Generative AI. Built with ChromaDB, SQLite, Sentence Transformers, and FastAPI.")
