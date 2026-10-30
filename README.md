# A Resource-Efficient Long-Term Memory Layer for Generative AI 🧠⚡

**Short Name:** `MemoryLayer AI`  
**Core Thesis:** *The memory layer does not replace the LLM. It supplies compact, relevant long-term context to the LLM while filtering out irrelevant noise and preventing context bloat.*

---

## 1. Problem
Standard Large Language Models (LLMs) operate statelessly. Each API invocation is independent, with zero persistent memory across sessions. To establish continuity across conversations, existing applications typically dump complete raw conversation histories into the LLM prompt window.

This naive approach causes severe architectural bottlenecks:
1. **Quadratic Cost Scaling**: Input token costs grow with every single turn, even when the user asks a simple general knowledge question.
2. **Context Window Exhaustion**: Long chats overflow token limits or trigger aggressive, naive truncation that discards valuable information.
3. **Context Pollution & Attention Degradation**: Irrelevant past chit-chat dilutes the LLM's attention mechanism ("Lost in the Middle" phenomenon), increasing hallucination risk.
4. **Latency Overhead**: Time-to-first-token increases proportionally with the size of the injected prompt context.

---

## 2. Proposed Solution
**MemoryLayer AI** introduces a dedicated, resource-efficient long-term memory layer that sits between the user and the Generative AI model. 

The memory layer is **NOT** the LLM itself; the LLM remains responsible for generating natural-language answers. Instead of passing complete conversation logs:
1. Significant facts and preferences are **extracted** into atomic memory nodes.
2. Redundant facts are **compressed** into high-density summaries.
3. Relevant memories are indexed in a **vector store (ChromaDB)** and **metadata store (SQLite)**.
4. When a query arrives, a **Relevance Gate** filters out weak or irrelevant memories.
5. A **Context Budget** strictly caps injected memory tokens.
6. Only **compact, highly relevant context** is provided to the LLM. If no memory is relevant or the query is general, the LLM receives the query alone.

```
                    USER
                      │
                      ▼
                 USER QUERY
                      │
                      ▼
             SEMANTIC RETRIEVAL
                      │
                      ▼
             RELEVANCE CHECK
                /           \
              YES            NO
               │              │
               ▼              ▼
       Relevant Memory     Query Only
               │              │
               ▼              │
        Context Optimization  │
               │              │
               └───────► LLM ◄┘
                         │
                         ▼
                  NATURAL ANSWER
                         │
                         ▼
                  MEMORY UPDATE
                         │
                         ▼
                KEEP / ARCHIVE / FORGET
```

---

## 3. Long-Term Memory
The memory layer manages multi-session persistent knowledge categorized into structured memory types:
- **`personal`**: Biographical user facts (e.g., name, education, profession, location).
- **`preference`**: Explicit user preferences (e.g., favorite programming languages, tools, cuisines).
- **`goal`**: User objectives, plans, and aspirations.
- **`project`**: Ongoing projects, architectural decisions, and technical stacks.
- **`fact`**: General user-stated factual statements.

Atomic memory nodes include intrinsic importance scores, confidence ratings, creation timestamps, access counts, and pin protection flags.

---

## 4. Memory Compression
Over time, repeated conversations create semantic clutter. The **Memory Compressor**:
1. Uses pairwise cosine similarity clustering to group related memories (e.g., multiple statements about Python, frameworks, and projects).
2. Synthesizes clusters into single, concise factual summary nodes.
3. Archives source memories while preserving full parent-child lineage in SQLite.
4. Measurably reduces character and token footprints by **50%–75%**, preserving essential facts while eliminating conversational filler.

---

## 5. Semantic Retrieval
Candidate retrieval balances vector similarity with semantic quality using a multi-factor re-ranking formula:
$$\text{final\_score} = 0.50 \times \text{similarity} + 0.20 \times \text{importance} + 0.15 \times \text{recency} + 0.15 \times \text{access\_frequency}$$

- **Dense Vector Search**: ChromaDB retrieves candidate nearest neighbors based on cosine similarity.
- **Semantic Overlap & Domain Associations**: Lexical and domain concepts (e.g., framework $\leftrightarrow$ FastAPI, education $\leftrightarrow$ B.Tech/CSE) ensure deterministic matching.
- **Recency Decay**: Exponential decay $e^{-\lambda \cdot \Delta t}$ prevents outdated memories from dominating.
- **Access Frequency Boost**: Frequently recalled facts receive a logarithmic access bonus.

---

## 6. LLM Integration
The LLM serves as the generative reasoning engine and receives strictly formatted, compact context:

- **Case A — General Knowledge Question**:
  - LLM receives query only (0 memories injected).
  - Answers normally using pre-trained knowledge base.
- **Case B — Relevant Personal Memory Exists**:
  - LLM receives query + compact filtered memory context.
  - Generates natural, second-person grounded response (e.g., *"Your name is Abhinandan."*).
- **Case C — Personal Question With No Stored Memory**:
  - LLM receives query without hallucinating unrecorded personal facts.
  - Clearly states that the information is not stored in active memory.
- **Case D — Mixed Question (Personal Context + Technical Concept)**:
  - LLM combines stored project context with general technical knowledge.
- **Case E — Unrelated Query With Stored Distractors**:
  - Distractor memories fail the relevance threshold and are excluded from the prompt.

Supported Providers:
- **OpenAI Provider**: Uses `gpt-4o-mini` (or user-configured model) with grounded system prompt.
- **Mock Provider**: Fully deterministic, zero-dependency pattern engine for fast offline testing and development.
- **Local HuggingFace Provider**: Extensible pipeline for local open-source models.

---

## 7. Resource Efficiency & Measurable Metrics
Instead of uncompressed conversation logs (~500 - 5,000+ tokens per turn), the system sends compact context (~20 - 60 tokens).

Real-time measurable resource metrics tracked on every chat turn:
- **`original_context_size` / `baseline_tokens` (EST.)**: Estimated tokens if all active user memories were dumped naively (~4 characters/token).
- **`injected_tokens` (EST.)**: Tokens consumed by the selected compact context.
- **`tokens_saved` (EST.)**: Difference between baseline and injected tokens.
- **`context_reduction_percent` (EST.)**: Measurable percentage context reduction (typically **75% – 95%** in multi-memory sessions).
- **`retrieval_latency_ms`**: Measured retrieval and re-ranking pipeline execution time (typically **< 5ms**).

*All token counts are labeled as estimates (EST.) when calculated without proprietary tokenizer binaries.*

---

## 8. Adaptive Retention Lifecycle: Keep, Archive, Forget
Memories evolve through an adaptive retention lifecycle:
- **KEEP (Active)**: High-importance, recent, or pinned memories available for retrieval.
- **ARCHIVE**: Stale or superseded memories preserved for audit and lineage.
- **FORGET**: Low-importance decayed memories automatically soft-deleted or pruned from vector indices.
- **PINNED Protection**: Critical memories (e.g., allergies, permanent identity, emergency contacts) marked as `is_pinned: True` are immune to decay pruning.

---

## 9. Technology Stack
- **Core Language**: Python 3.10+
- **Vector Database**: ChromaDB (Embedded vector store with cosine distance)
- **Relational Metadata Database**: SQLite (Thread-safe, schema migrations, foreign keys)
- **Embeddings**: HuggingFace Sentence Transformers (`all-MiniLM-L6-v2`, 384-dimensional dense vectors)
- **Backend API**: FastAPI + Pydantic v2 + Uvicorn
- **User Interface**: Streamlit (Reactive dark-mode dashboard)
- **Machine Learning**: PyTorch Contrastive Projection Network for task-adaptive embedding alignment

---

## 10. How to Run the Project

### 1. Prerequisites
Ensure Python 3.10+ is installed.

### 2. Environment Setup
```powershell
# Navigate into the project folder
cd c:\Users\abhin\Desktop\projects\ai-memory-system

# Activate virtual environment
.\.venv\Scripts\Activate.ps1
```

### 3. Launch Streamlit UI
```powershell
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

The dashboard contains 5 dedicated sections:
1. **💬 AI Assistant**: Natural chat interface with real-time memory injection badges, context token tracking, and 1-click demo scenario buttons.
2. **🧠 Memory Explorer**: Full lifecycle table (Active / Archived / Forgotten) with manual creation and archive/reactivate/delete actions.
3. **🔎 Semantic Retrieval**: Diagnostic sandbox showing query $\rightarrow$ vector candidates $\rightarrow$ composite score $\rightarrow$ relevance gate decisions.
4. **⚡ Resource Efficiency**: Live KPI cards, architectural comparison breakdown (Traditional vs. MemoryLayer AI), and semantic clustering tools.
5. **📊 System Analytics**: Database statistics, ChromaDB vector counts, and decay pruning controls.

### 4. Launch FastAPI Backend (Optional)
```powershell
uvicorn backend.main:app --reload --port 8000
```
Interactive OpenAPI documentation is available at `http://localhost:8000/docs`.

---

## 11. Testing
Run the complete automated test suite:
```powershell
pytest tests/ -v
```

Current test status:
- **71 passed, 0 failures, 3 warnings** across all 10 test modules.
- Tests cover:
  1. General questions with zero memory injection
  2. Relevant personal memory extraction and retrieval
  3. Irrelevant memory filtering and distractor suppression
  4. Non-hallucination of unrecorded personal attributes
  5. Semantic similarity and domain concept matching
  6. Memory compression and cluster summarization
  7. Top-K limit enforcement and context budgeting
  8. Composite threshold gating
  9. Real-time resource efficiency metric calculations
  10. Keep / Archive / Forget lifecycle and pinned item protection

---

## 12. Example User Flows

### Example 1: General Knowledge Question
- **User**: *"What is the national animal of India?"*
- **Retrieval**: 0 memories relevant $\rightarrow$ Relevance Gate filters out distractor memories.
- **Context to LLM**: Query only (`context=None`).
- **Response**: *"The national animal of India is the Bengal Tiger (Panthera tigris)..."*

### Example 2: Storing and Recalling Personal Identity
- **Turn 1 — Store**:
  - **User**: *"My name is Abhinandan."*
  - **Memory Extracted**: `User's name is Abhinandan.` (Type: `personal`, Importance: `0.95`).
  - **Response**: *"Nice to meet you, Abhinandan! I've saved that to my memory."*
- **Turn 2 — Recall**:
  - **User**: *"What is my name?"*
  - **Retrieval**: High semantic similarity $\rightarrow$ Score $\ge 0.65$ $\rightarrow$ Gate: `relevant`.
  - **Response**: *"Your name is Abhinandan."*

### Example 3: Unrecorded Attribute (Zero Hallucination)
- **User**: *"What is my favorite programming language?"* (no memory stored yet)
- **Retrieval**: No matching preference stored $\rightarrow$ Gate: rejected.
- **Response**: *"I don't have any specific active memories on that yet. Tell me something about yourself and I will remember it!"*

### Example 4: Unrelated Query with Stored Distractor Memories
- **Stored Memory**: `User's name is Abhinandan.`
- **User**: *"What is the capital of France?"*
- **Retrieval**: Relevance score for name memory is 0.34 < 0.65 $\rightarrow$ Gate: `filtered out`.
- **Response**: *"The capital of France is Paris."* (Name memory is never mentioned or injected).

### Example 5: Storing and Recalling Explicit Preferences
- **Turn 1 — Store**:
  - **User**: *"My favorite programming language is Python."*
  - **Memory Extracted**: `User's favorite programming language is Python.` (Type: `preference`, Importance: `0.90`).
  - **Response**: *"Got it! I've saved to my memory that your favorite programming language is Python."*
- **Turn 2 — Recall**:
  - **User**: *"What is my favorite programming language?"*
  - **Retrieval**: Re-ranking score 0.83 $\ge 0.65$ $\rightarrow$ Injected into compact context.
  - **Response**: *"Your favorite programming language is Python."*
