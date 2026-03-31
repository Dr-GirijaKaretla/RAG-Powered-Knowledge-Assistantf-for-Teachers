"""
📚 Teacher's AI Knowledge Assistant — Streamlit Application

Multi-page RAG web app that lets teachers upload documents, ask
questions, generate quizzes, and summarise content.  All processing
runs 100 % locally via HuggingFace models.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st

# ──────────────────────────────────────────────────────────────
# Page config (MUST be the first Streamlit command)
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Teacher's AI Knowledge Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

from src.evaluation.metrics import EvaluationMetrics
from src.features.quiz_generator import QuizGenerator
from src.pipeline.rag_pipeline import RAGPipeline
from src.utils.visualizer import Visualizer

# ──────────────────────────────────────────────────────────────
#  Custom CSS
# ──────────────────────────────────────────────────────────────

_CUSTOM_CSS = """
<style>
/* ── Global ────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── Chat bubbles ─────────────────────────── */
.user-bubble {
    background: linear-gradient(135deg, #6366F1, #8B5CF6);
    color: white; padding: 12px 18px; border-radius: 16px 16px 4px 16px;
    margin: 6px 0; max-width: 85%;
}
.assistant-bubble {
    background: #F1F5F9; color: #1E293B; padding: 12px 18px;
    border-radius: 16px 16px 16px 4px; margin: 6px 0; max-width: 85%;
    border-left: 4px solid #6366F1;
}

/* ── Source citation pill ─────────────────── */
.source-pill {
    display: inline-block; background: #EEF2FF; color: #4338CA;
    padding: 2px 10px; border-radius: 12px; font-size: 0.75rem;
    margin: 2px 4px; font-weight: 500;
}

/* ── Feature cards ────────────────────────── */
.feature-card {
    background: white; border-radius: 16px; padding: 28px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06); text-align: center;
    transition: transform 0.2s, box-shadow 0.2s;
    border: 1px solid #E2E8F0;
}
.feature-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 8px 24px rgba(99,102,241,0.15);
}

/* ── Quiz cards ───────────────────────────── */
.quiz-card {
    background: white; border-left: 5px solid #6366F1;
    border-radius: 12px; padding: 20px; margin: 12px 0;
    box-shadow: 0 1px 6px rgba(0,0,0,0.05);
}

/* ── Metric cards ─────────────────────────── */
.metric-card {
    background: linear-gradient(135deg, #6366F1 0%, #8B5CF6 100%);
    color: white; border-radius: 14px; padding: 20px; text-align: center;
}
.metric-card h3 { margin: 0; font-size: 2rem; font-weight: 700; }
.metric-card p  { margin: 4px 0 0 0; opacity: 0.85; font-size: 0.85rem; }

/* ── Status indicators ────────────────────── */
.status-ready  { color: #10B981; font-weight: 600; }
.status-empty  { color: #EF4444; font-weight: 600; }

/* ── Misc ─────────────────────────────────── */
.step-box {
    background: #F8FAFC; border-radius: 12px; padding: 20px;
    text-align: center; border: 1px solid #E2E8F0;
}
</style>
"""

st.markdown(_CUSTOM_CSS, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
#  Session state initialisation
# ──────────────────────────────────────────────────────────────

def _init_session_state() -> None:
    """Ensure all required session-state keys exist."""
    defaults: Dict[str, Any] = {
        "pipeline": None,
        "chat_history": [],
        "uploaded_docs": [],
        "current_answer": None,
        "current_quiz": None,
        "active_page": "🏠 Home",
        "session_start": time.time(),
        "questions_asked": 0,
        "docs_uploaded": 0,
        "quizzes_generated": 0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


_init_session_state()


# ──────────────────────────────────────────────────────────────
#  Pipeline initialisation (cached)
# ──────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Initialising RAG pipeline …")
def _get_pipeline() -> RAGPipeline:
    """Instantiate and cache the RAG pipeline."""
    return RAGPipeline(config_path="configs/config.yaml")


def _ensure_pipeline() -> RAGPipeline:
    """Return the cached pipeline and store in session state."""
    if st.session_state.pipeline is None:
        st.session_state.pipeline = _get_pipeline()
    return st.session_state.pipeline


# ──────────────────────────────────────────────────────────────
#  Sidebar
# ──────────────────────────────────────────────────────────────

def _render_sidebar() -> str:
    """Draw the sidebar and return the selected page name."""
    with st.sidebar:
        st.markdown("## 📚 Knowledge Assistant")
        st.caption("AI-powered teaching companion")
        st.divider()

        page = st.radio(
            "Navigation",
            [
                "🏠 Home",
                "📤 Document Management",
                "💬 Ask Questions",
                "📄 Document Summarizer",
                "📝 Quiz Generator",
                "📊 Analytics & Evaluation",
                "ℹ️ About & Help",
            ],
            key="nav_radio",
        )

        st.divider()

        # Knowledge base status
        try:
            pipeline = _ensure_pipeline()
            docs = pipeline.list_documents()
            if docs:
                st.markdown(f'<span class="status-ready">🟢 Ready — {len(docs)} doc(s)</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="status-empty">🔴 No documents loaded</span>', unsafe_allow_html=True)
        except Exception:
            st.markdown('<span class="status-empty">🟡 Initialising …</span>', unsafe_allow_html=True)

        # Model info expander
        with st.expander("🤖 Model Info"):
            try:
                pipeline = _ensure_pipeline()
                stats = pipeline.get_system_stats()
                emb = stats.get("embedding_model", {})
                gen = stats.get("generation_model", {})
                st.markdown(f"**Embedder:** {emb.get('model_name', 'N/A')}")
                st.markdown(f"**Dim:** {emb.get('embedding_dim', 'N/A')}")
                st.markdown(f"**LLM:** {gen.get('model_name', 'N/A')}")
            except Exception:
                st.info("Pipeline not ready yet.")

        st.divider()

        # Clear everything
        if st.button("🗑️ Clear Everything", use_container_width=True):
            try:
                pipeline = _ensure_pipeline()
                pipeline.vectorstore.reset_collection(confirm=True)
                st.session_state.chat_history = []
                st.session_state.uploaded_docs = []
                st.session_state.current_answer = None
                st.session_state.current_quiz = None
                st.session_state.questions_asked = 0
                st.session_state.docs_uploaded = 0
                st.session_state.quizzes_generated = 0
                st.success("All data cleared!")
                st.rerun()
            except Exception as exc:
                st.error(f"Error: {exc}")

        # Session timer
        elapsed = time.time() - st.session_state.session_start
        mins = int(elapsed // 60)
        secs = int(elapsed % 60)
        st.caption(f"⏱️ Session: {mins}m {secs}s")

    return page


# ══════════════════════════════════════════════════════════════
#  PAGE 1: 🏠 Home
# ══════════════════════════════════════════════════════════════

def _page_home() -> None:
    """Render the landing page."""
    st.markdown("# 📚 Teacher's AI Knowledge Assistant")
    st.markdown("### *Your AI-powered teaching companion*")
    st.markdown("Upload your educational materials, ask questions, generate quizzes — all powered by AI running **100% locally**.")

    st.divider()

    # Feature cards
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            '<div class="feature-card">'
            '<h2>📤</h2><h4>Upload Materials</h4>'
            '<p>PDF, DOCX, TXT — your documents are parsed, chunked, and indexed automatically.</p>'
            '</div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            '<div class="feature-card">'
            '<h2>💬</h2><h4>Ask Questions</h4>'
            '<p>Get accurate, grounded answers with cited sources from your own materials.</p>'
            '</div>',
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            '<div class="feature-card">'
            '<h2>📝</h2><h4>Generate Quizzes</h4>'
            '<p>Auto-create multiple-choice quizzes from any topic or document.</p>'
            '</div>',
            unsafe_allow_html=True,
        )

    st.divider()

    # How it works
    st.markdown("### 🔄 How It Works")
    s1, s2, s3, s4 = st.columns(4)
    steps = [
        ("1️⃣", "Upload", "Upload PDF, DOCX, or TXT files."),
        ("2️⃣", "Index", "Documents are chunked and embedded."),
        ("3️⃣", "Ask", "Ask natural-language questions."),
        ("4️⃣", "Answer", "Get grounded, cited answers."),
    ]
    for col, (icon, title, desc) in zip([s1, s2, s3, s4], steps):
        with col:
            st.markdown(
                f'<div class="step-box">'
                f'<h2>{icon}</h2><h4>{title}</h4><p style="font-size:0.85rem">{desc}</p>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.divider()

    # System status
    st.markdown("### 📡 System Status")
    try:
        pipeline = _ensure_pipeline()
        stats = pipeline.get_system_stats()
        vs = stats["vectorstore"]
        m1, m2, m3 = st.columns(3)
        m1.metric("Documents", vs["total_documents"])
        m2.metric("Total Chunks", vs["total_chunks"])
        m3.metric("Pipeline", "✅ Ready")
    except Exception:
        st.warning("Pipeline is still loading. Please wait …")

    st.divider()

    # Demo content loader + Quick start
    col_demo, col_quick = st.columns(2)
    with col_demo:
        if st.button("📦 Load Demo Content", use_container_width=True):
            try:
                pipeline = _ensure_pipeline()
                with st.spinner("Loading demo documents …"):
                    reports = pipeline.ingest_demo_content()
                if reports:
                    st.success(f"Loaded {len(reports)} demo document(s)!")
                    for r in reports:
                        st.write(f"  ✅ **{r['filename']}** — {r['chunks_stored']} chunks")
                    st.session_state.docs_uploaded += len(reports)
                else:
                    st.info("Demo documents already loaded.")
                st.rerun()
            except Exception as exc:
                st.error(f"Failed to load demos: {exc}")

    with col_quick:
        if st.button("📤 Go to Document Upload →", use_container_width=True):
            st.session_state.nav_radio = "📤 Document Management"
            st.rerun()


# ══════════════════════════════════════════════════════════════
#  PAGE 2: 📤 Document Management
# ══════════════════════════════════════════════════════════════

def _page_document_management() -> None:
    """Render the document upload and management page."""
    st.markdown("# 📤 Document Management")

    pipeline = _ensure_pipeline()

    # ── Upload panel ──────────────────────────────────────────
    st.markdown("### Upload Documents")
    uploaded_files = st.file_uploader(
        "Choose PDF, DOCX, or TXT files",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        key="file_uploader",
    )

    if uploaded_files and st.button("📥 Upload & Index", use_container_width=True):
        for file in uploaded_files:
            try:
                # Validate first
                validation = pipeline.parser.validate(file)
                if not validation["valid"]:
                    st.error(f"❌ {file.name}: {validation['error']}")
                    continue

                # Check for duplicates
                if file.name in pipeline.list_documents():
                    st.warning(f"⚠️ '{file.name}' already exists — skipping. Delete it first to re-upload.")
                    continue

                progress_bar = st.progress(0, text="Starting …")

                def update_progress(frac: float, msg: str) -> None:
                    progress_bar.progress(frac, text=msg)

                report = pipeline.ingest_document(file, progress_callback=update_progress)
                progress_bar.empty()

                # Show report
                st.success(f"✅ **{report['filename']}** indexed successfully!")
                rc1, rc2, rc3 = st.columns(3)
                rc1.metric("Pages Parsed", report["pages_parsed"])
                rc2.metric("Chunks Created", report["chunks_stored"])
                rc3.metric("Time", f"{report['ingestion_time_ms']:.0f} ms")

                st.session_state.docs_uploaded += 1

            except Exception as exc:
                st.error(f"❌ Error processing '{file.name}': {exc}")

    st.divider()

    # ── Knowledge Base Manager ────────────────────────────────
    st.markdown("### 📂 Knowledge Base")

    docs = pipeline.list_documents()
    if not docs:
        st.info("No documents in the knowledge base yet. Upload files above or load demo content from the Home page.")
        return

    stats = pipeline.vectorstore.get_collection_stats()
    km1, km2 = st.columns(2)
    km1.metric("Total Documents", stats["total_documents"])
    km2.metric("Total Chunks", stats["total_chunks"])

    st.markdown("---")

    for doc_name in docs:
        chunks = pipeline.vectorstore.get_document_chunks(doc_name)
        pages = set(c.get("page", 0) for c in chunks)

        with st.container():
            dc1, dc2, dc3, dc4, dc5 = st.columns([3, 1, 1, 1, 1])
            dc1.markdown(f"**📄 {doc_name}**")
            ext = Path(doc_name).suffix.lstrip(".")
            dc2.caption(ext.upper())
            dc3.caption(f"{len(chunks)} chunks")
            dc4.caption(f"{len(pages)} page(s)")

            with dc5:
                if st.button("🗑️", key=f"del_{doc_name}", help=f"Delete {doc_name}"):
                    try:
                        deleted = pipeline.delete_document(doc_name)
                        st.success(f"Deleted {deleted} chunks from '{doc_name}'.")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Error deleting: {exc}")

    st.divider()
    if st.button("🗑️ Clear All Documents", use_container_width=True):
        pipeline.vectorstore.reset_collection(confirm=True)
        st.success("All documents cleared!")
        st.rerun()


# ══════════════════════════════════════════════════════════════
#  PAGE 3: 💬 Ask Questions
# ══════════════════════════════════════════════════════════════

def _page_ask_questions() -> None:
    """Render the main RAG chat interface."""
    st.markdown("# 💬 Ask Questions")

    pipeline = _ensure_pipeline()
    docs = pipeline.list_documents()

    if not docs:
        st.warning("⚠️ No documents loaded. Please upload documents first or load demo content.")
        return

    # Layout
    left_col, right_col = st.columns([7, 3])

    with left_col:
        # Clear chat
        if st.button("🗑️ Clear Chat", key="clear_chat"):
            st.session_state.chat_history = []
            st.session_state.current_answer = None
            st.rerun()

        # Chat history
        for msg in st.session_state.chat_history:
            role = msg.get("role", "user")
            with st.chat_message(role):
                st.markdown(msg["content"])
                if role == "assistant" and msg.get("sources"):
                    with st.expander("📎 Sources"):
                        for src in msg["sources"]:
                            st.markdown(
                                f'<span class="source-pill">{src["source"]} • Page {src["page"]} • Score: {src["score"]:.3f}</span>',
                                unsafe_allow_html=True,
                            )

        # Chat input
        question = st.chat_input("Ask a question about your materials …")
        if question:
            # Append user message
            st.session_state.chat_history.append({"role": "user", "content": question})

            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):
                with st.spinner("Searching knowledge base …"):
                    try:
                        result = pipeline.ask(
                            question,
                            chat_history=st.session_state.chat_history,
                        )
                        answer = result["answer"]
                        sources = result.get("sources", [])

                        st.markdown(answer)
                        if sources:
                            with st.expander("📎 Sources"):
                                for src in sources:
                                    st.markdown(
                                        f'<span class="source-pill">{src["source"]} • Page {src["page"]} • Score: {src["score"]:.3f}</span>',
                                        unsafe_allow_html=True,
                                    )

                        # Store in history
                        st.session_state.chat_history.append(
                            {"role": "assistant", "content": answer, "sources": sources}
                        )
                        st.session_state.current_answer = result
                        st.session_state.questions_asked += 1

                        # Trim history
                        max_hist = pipeline.config.get("app", {}).get("max_chat_history", 20)
                        if len(st.session_state.chat_history) > max_hist * 2:
                            st.session_state.chat_history = st.session_state.chat_history[-(max_hist * 2):]

                    except Exception as exc:
                        st.error(f"Error generating answer: {exc}")

    with right_col:
        st.markdown("### 🔍 Retrieval Inspector")
        result = st.session_state.current_answer

        if result:
            # Similarity chart
            retrieval_results = result.get("retrieval_results", [])
            if retrieval_results:
                fig = Visualizer.similarity_scores_chart(retrieval_results)
                st.plotly_chart(fig, use_container_width=True)

            # Top chunks
            st.markdown("**Top Retrieved Chunks:**")
            for i, r in enumerate(retrieval_results[:3]):
                with st.expander(f"Chunk {i+1}: {r.get('source', '?')} (p{r.get('page', '?')})"):
                    st.text(r.get("text", "")[:300] + " …" if len(r.get("text", "")) > 300 else r.get("text", ""))
                    score = r.get("rerank_score") or r.get("similarity_score", 0)
                    st.caption(f"Score: {score:.4f}")

            # Query stats
            st.markdown("---")
            qs1, qs2 = st.columns(2)
            qs1.metric("Retrieval", f"{result.get('retrieval_time_ms', 0):.0f} ms")
            qs2.metric("Generation", f"{result.get('generation_time_ms', 0):.0f} ms")
        else:
            st.info("Ask a question to see retrieval details.")

        # Settings expander
        with st.expander("⚙️ Settings"):
            st.slider("Top-K", 1, 10, pipeline.retriever.top_k, key="topk_slider")
            st.toggle("Re-rank", value=pipeline.retriever.use_rerank, key="rerank_toggle")
            st.slider("Sim. Threshold", 0.0, 1.0, pipeline.retriever.similarity_threshold, 0.05, key="threshold_slider")


# ══════════════════════════════════════════════════════════════
#  PAGE 4: 📄 Document Summarizer
# ══════════════════════════════════════════════════════════════

def _page_summarizer() -> None:
    """Render the document summarisation page."""
    st.markdown("# 📄 Document Summarizer")

    pipeline = _ensure_pipeline()
    docs = pipeline.list_documents()

    if not docs:
        st.warning("No documents available. Upload documents first.")
        return

    selected = st.selectbox("Select a document to summarize", docs)

    if st.button("📝 Generate Summary", use_container_width=True):
        with st.spinner("Reading document … Generating summary …"):
            try:
                result = pipeline.summarize(selected)

                # Compression stats
                st.markdown("### 📊 Compression Stats")
                cs1, cs2, cs3 = st.columns(3)
                cs1.metric("Original Words", result["word_count_original"])
                cs2.metric("Summary Words", result["word_count_summary"])
                cs3.metric("Compression Ratio", f"{result['compression_ratio']}×")

                st.divider()

                # Summary
                st.markdown("### 📝 Summary")
                st.markdown(
                    f'<div style="background:#F8FAFC; border-radius:12px; padding:20px; '
                    f'border-left:4px solid #6366F1;">{result["summary"]}</div>',
                    unsafe_allow_html=True,
                )

                # Download
                st.divider()
                st.download_button(
                    "📥 Download Summary (.txt)",
                    data=result["summary"],
                    file_name=f"summary_{selected}.txt",
                    mime="text/plain",
                    use_container_width=True,
                )
            except Exception as exc:
                st.error(f"Error generating summary: {exc}")


# ══════════════════════════════════════════════════════════════
#  PAGE 5: 📝 Quiz Generator
# ══════════════════════════════════════════════════════════════

def _page_quiz_generator() -> None:
    """Render the quiz generation page."""
    st.markdown("# 📝 Quiz Generator")

    pipeline = _ensure_pipeline()
    docs = pipeline.list_documents()

    if not docs:
        st.warning("No documents available. Upload documents first.")
        return

    # Input method
    method = st.radio("Generate quiz from:", ["By Topic", "By Document"], horizontal=True)

    topic_input = None
    doc_input = None

    if method == "By Topic":
        topic_input = st.text_input("Enter topic (e.g., 'photosynthesis')")
    else:
        doc_input = st.selectbox("Select document", docs)

    num_q = st.slider("Number of questions", 3, 10, 5)
    difficulty = st.select_slider("Difficulty", ["easy", "medium", "hard"], value="medium")

    if st.button("🎯 Generate Quiz", use_container_width=True):
        with st.spinner("Crafting questions …"):
            try:
                if method == "By Topic" and topic_input:
                    quiz = pipeline.generate_quiz(topic=topic_input, num_questions=num_q, difficulty=difficulty)
                elif method == "By Document" and doc_input:
                    quiz = pipeline.generate_quiz(source_filename=doc_input, num_questions=num_q, difficulty=difficulty)
                else:
                    st.warning("Please provide a topic or select a document.")
                    return

                if not quiz:
                    st.warning("Could not generate quiz questions. Try a different topic or document.")
                    return

                st.session_state.current_quiz = quiz
                st.session_state.quizzes_generated += 1

            except Exception as exc:
                st.error(f"Error generating quiz: {exc}")
                return

    # Display quiz
    quiz = st.session_state.current_quiz
    if quiz:
        st.divider()
        st.markdown(f"### 📋 Quiz ({len(quiz)} Questions)")

        display_quiz = QuizGenerator.format_quiz_for_display(quiz)

        for item in display_quiz:
            st.markdown(
                f'<div class="quiz-card"><strong>Q{item["number"]}: {item["question"]}</strong></div>',
                unsafe_allow_html=True,
            )

            options = item["options"]
            user_answer = st.radio(
                f"Select your answer for Q{item['number']}:",
                [f"{k}) {v}" for k, v in options.items() if v],
                key=f"quiz_q_{item['number']}",
                label_visibility="collapsed",
            )

            if st.button(f"✅ Check Answer Q{item['number']}", key=f"check_{item['number']}"):
                correct = item["answer"]
                if user_answer and user_answer.startswith(correct):
                    st.success(f"✅ Correct! The answer is {correct}.")
                else:
                    st.error(f"❌ Incorrect. The correct answer is {correct}.")
                if item["explanation"]:
                    st.info(f"💡 **Explanation:** {item['explanation']}")

            st.markdown("---")

        # Bottom actions
        ba1, ba2 = st.columns(2)
        with ba1:
            quiz_text = QuizGenerator.export_quiz_as_text(quiz)
            st.download_button(
                "📥 Download Quiz as TXT",
                data=quiz_text,
                file_name="quiz_export.txt",
                mime="text/plain",
                use_container_width=True,
            )
        with ba2:
            if st.button("🔄 Regenerate", use_container_width=True):
                st.session_state.current_quiz = None
                st.rerun()


# ══════════════════════════════════════════════════════════════
#  PAGE 6: 📊 Analytics & Evaluation
# ══════════════════════════════════════════════════════════════

def _page_analytics() -> None:
    """Render the analytics and evaluation page."""
    st.markdown("# 📊 Analytics & Evaluation")

    pipeline = _ensure_pipeline()
    vis = Visualizer()
    metrics_engine = EvaluationMetrics()

    # ── Knowledge Base Overview ───────────────────────────────
    st.markdown("### 📚 Knowledge Base Overview")
    stats = pipeline.vectorstore.get_collection_stats()
    ao1, ao2, ao3 = st.columns(3)
    ao1.metric("Documents", stats["total_documents"])
    ao2.metric("Total Chunks", stats["total_chunks"])
    ao3.metric("Questions Asked", st.session_state.questions_asked)

    docs = pipeline.list_documents()
    if docs:
        # Chunk map
        fig_map = vis.document_chunk_map(pipeline.vectorstore)
        st.plotly_chart(fig_map, use_container_width=True)

        # Chunk size distribution (from first doc as sample)
        all_chunks: List[Dict[str, Any]] = []
        for doc in docs:
            all_chunks.extend(pipeline.vectorstore.get_document_chunks(doc))
        if all_chunks:
            fig_dist = vis.chunk_size_distribution(all_chunks)
            st.plotly_chart(fig_dist, use_container_width=True)

    st.divider()

    # ── Retrieval Quality ─────────────────────────────────────
    st.markdown("### 🎯 Retrieval Quality")
    result = st.session_state.current_answer
    if result:
        rq1, rq2 = st.columns(2)
        with rq1:
            retrieval_results = result.get("retrieval_results", [])
            if retrieval_results:
                fig_sim = vis.similarity_scores_chart(retrieval_results)
                st.plotly_chart(fig_sim, use_container_width=True)

        with rq2:
            # Answer relevance
            try:
                relevance = metrics_engine.answer_relevance_score(
                    result["question"], result["answer"], pipeline.embedder
                )
                st.metric("Answer Relevance", f"{relevance:.1%}")
            except Exception:
                st.metric("Answer Relevance", "N/A")

            # Context faithfulness
            try:
                faithfulness = metrics_engine.context_faithfulness(
                    result["answer"], result.get("context_used", "")
                )
                st.metric("Context Faithfulness", f"{faithfulness:.1%}")
            except Exception:
                st.metric("Context Faithfulness", "N/A")

        # Timing
        fig_time = metrics_engine.display_answer_stats(result)
        st.plotly_chart(fig_time, use_container_width=True)

        # Timeline
        fig_wf = vis.retrieval_timeline(result)
        st.plotly_chart(fig_wf, use_container_width=True)
    else:
        st.info("Ask a question on the 'Ask Questions' page to see retrieval quality metrics.")

    st.divider()

    # ── Session Statistics ────────────────────────────────────
    st.markdown("### 📈 Session Statistics")
    ss1, ss2, ss3, ss4 = st.columns(4)
    ss1.metric("Questions Asked", st.session_state.questions_asked)
    ss2.metric("Docs Uploaded", st.session_state.docs_uploaded)
    ss3.metric("Quizzes Generated", st.session_state.quizzes_generated)

    elapsed = time.time() - st.session_state.session_start
    ss4.metric("Session Duration", f"{int(elapsed // 60)}m {int(elapsed % 60)}s")


# ══════════════════════════════════════════════════════════════
#  PAGE 7: ℹ️ About & Help
# ══════════════════════════════════════════════════════════════

def _page_about() -> None:
    """Render the about and help page."""
    st.markdown("# ℹ️ About & Help")

    # Architecture diagram
    st.markdown("### 🏗️ System Architecture")
    st.graphviz_chart("""
        digraph RAG {
            rankdir=LR;
            node [shape=box, style="rounded,filled", fontname="Inter"];

            Upload  [label="📤 Upload", fillcolor="#EEF2FF"];
            Parse   [label="📄 Parse",  fillcolor="#EEF2FF"];
            Clean   [label="🧹 Clean",  fillcolor="#EEF2FF"];
            Chunk   [label="✂️ Chunk",  fillcolor="#EEF2FF"];
            Embed   [label="🔢 Embed",  fillcolor="#DBEAFE"];
            Store   [label="💾 Store",  fillcolor="#DBEAFE"];
            Query   [label="❓ Query",  fillcolor="#FEF3C7"];
            Retrieve[label="🔍 Retrieve", fillcolor="#FEF3C7"];
            Rerank  [label="🏆 Rerank", fillcolor="#FEF3C7"];
            Generate[label="🤖 Generate", fillcolor="#D1FAE5"];
            Answer  [label="✅ Answer", fillcolor="#D1FAE5"];

            Upload -> Parse -> Clean -> Chunk -> Embed -> Store;
            Query -> Embed;
            Embed -> Retrieve -> Rerank -> Generate -> Answer;
            Store -> Retrieve;
        }
    """)

    st.divider()

    # Model info
    st.markdown("### 🤖 Model Information")
    try:
        pipeline = _ensure_pipeline()
        stats = pipeline.get_system_stats()
        emb = stats.get("embedding_model", {})
        gen = stats.get("generation_model", {})

        model_data = {
            "Component": ["Embedding Model", "LLM (Generation)", "Re-ranker"],
            "Model": [
                emb.get("model_name", "N/A"),
                gen.get("model_name", "N/A"),
                pipeline.retriever.reranker_model_name,
            ],
            "Details": [
                f"Dim: {emb.get('embedding_dim', 'N/A')} | Device: {emb.get('device', 'N/A')}",
                f"Max tokens: {gen.get('max_new_tokens', 'N/A')}",
                f"Re-rank enabled: {pipeline.retriever.use_rerank}",
            ],
        }
        st.table(model_data)
    except Exception:
        st.info("Pipeline not ready — model info will appear once loaded.")

    st.divider()

    # FAQ
    st.markdown("### ❓ Frequently Asked Questions")

    with st.expander("What file formats are supported?"):
        st.markdown("The assistant supports **PDF**, **DOCX**, and **TXT** files up to 20 MB each.")

    with st.expander("How does RAG work?"):
        st.markdown(
            "**Retrieval-Augmented Generation (RAG)** combines a search engine with a language model. "
            "Your documents are split into chunks and embedded as vectors. When you ask a question, "
            "the most relevant chunks are retrieved and fed to the LLM as context, ensuring the answer "
            "is grounded in your actual materials."
        )

    with st.expander("Why does the AI say it can't find information?"):
        st.markdown(
            "This means the relevant information was not found among the retrieved chunks. Try:\n"
            "- Rephrasing your question\n"
            "- Uploading a document that covers the topic\n"
            "- Increasing the Top-K setting in the Ask Questions page"
        )

    with st.expander("How many documents can I upload?"):
        st.markdown(
            "There is no hard limit on the number of documents. However, more documents mean "
            "more chunks to search, which may slightly increase retrieval time."
        )

    with st.expander("How are chunks created?"):
        st.markdown(
            "Documents are split using a **recursive character text splitter** that tries to break "
            "on natural boundaries (paragraphs, sentences, spaces). Default chunk size is 512 characters "
            "with 64 characters of overlap to preserve context across boundaries."
        )

    st.divider()

    # Tech stack
    st.markdown("### 🛠️ Technology Stack")
    st.markdown(
        "| Component | Technology |\n"
        "|---|---|\n"
        "| UI | Streamlit |\n"
        "| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |\n"
        "| Vector Store | ChromaDB (persistent) |\n"
        "| LLM | HuggingFace (google/flan-t5-large) |\n"
        "| PDF Parsing | pdfplumber |\n"
        "| DOCX Parsing | python-docx |\n"
        "| Re-ranking | cross-encoder/ms-marco-MiniLM-L-6-v2 |\n"
        "| Visualization | Plotly |\n"
    )

    st.divider()

    # Tips
    st.markdown("### 💡 Tips for Best Results")
    st.markdown(
        "1. **Upload focused documents** — subject-specific materials yield better answers.\n"
        "2. **Ask specific questions** — instead of 'tell me about biology', try 'what is the Calvin cycle?'\n"
        "3. **Use the summarizer first** — get an overview before diving into detailed questions.\n"
        "4. **Check sources** — always verify the cited sources in the answer.\n"
        "5. **Adjust Top-K** — increase Top-K for broader questions, decrease for precise lookups."
    )


# ══════════════════════════════════════════════════════════════
#  Main routing
# ══════════════════════════════════════════════════════════════

def main() -> None:
    """Application entry point — render sidebar and active page."""
    page = _render_sidebar()
    st.session_state.active_page = page

    page_map = {
        "🏠 Home": _page_home,
        "📤 Document Management": _page_document_management,
        "💬 Ask Questions": _page_ask_questions,
        "📄 Document Summarizer": _page_summarizer,
        "📝 Quiz Generator": _page_quiz_generator,
        "📊 Analytics & Evaluation": _page_analytics,
        "ℹ️ About & Help": _page_about,
    }

    renderer = page_map.get(page, _page_home)
    renderer()


if __name__ == "__main__":
    main()
