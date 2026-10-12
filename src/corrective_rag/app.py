"""Streamlit chat UI for the Corrective RAG pipeline."""

import os

import streamlit as st

from corrective_rag.pipeline import DATA_DIR, build_graph, build_vector_store, doc_label

st.set_page_config(page_title="Corrective RAG", page_icon=":material/manage_search:")

SUGGESTIONS = {
    ":material/neurology: Scaled attention": "Why is dot-product attention scaled by 1/sqrt(d_k) in the Transformer?",
    ":material/memory: RAG memory": "How does RAG combine parametric and non-parametric memory?",
    ":material/compare_arrows: Self-RAG vs CRAG": "How do Self-RAG and Corrective RAG differ in handling irrelevant retrieved documents?",
    ":material/travel_explore: Off-corpus": "How does FlashAttention reduce the memory cost of attention?",
}


@st.cache_resource(show_spinner="Loading the paper index (the first run embeds all PDFs)...")
def load_graph():
    return build_graph(build_vector_store())


def render_assistant(msg: dict, key: str) -> None:
    """Replay a past assistant answer: pipeline steps, answer, sources."""
    with st.expander(msg["steps_label"], type="compact", key=f"steps_{key}"):
        for step in msg["steps"]:
            st.markdown(f"- {step}")
    st.markdown(msg["content"])
    if msg["sources"]:
        with st.expander("Sources", icon=":material/menu_book:", key=f"sources_{key}"):
            for source in msg["sources"]:
                st.markdown(f"- {source}")


def format_source(doc) -> str:
    label = doc_label(doc)
    return f"[{doc.metadata.get('title') or label}]({label})" if label.startswith("http") else label


# ------------------------------------------------------------------- sidebar

with st.sidebar:
    st.header("Corrective RAG")
    st.caption(
        "Answers questions from the indexed papers. Each retrieved chunk is graded for relevance; "
        "if none pass, the question is rewritten and sent to a web search instead."
    )
    st.subheader("Indexed papers")
    for pdf in sorted(DATA_DIR.glob("*.pdf")):
        st.markdown(f"- `{pdf.name}`")
    if st.button("Clear chat", icon=":material/delete:", width="stretch"):
        st.session_state.messages = []
        st.rerun()

# ---------------------------------------------------------------------- chat

st.title("Ask the papers")

if "messages" not in st.session_state:
    st.session_state.messages = []

if not os.getenv("GROQ_API_KEY"):
    st.error("`GROQ_API_KEY` is not set. Add it to `.env` (or pass it with `--env-file .env` in Docker).")
    st.stop()

for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(msg["content"])
        else:
            render_assistant(msg, str(i))

prompt = st.chat_input("Ask about attention, RAG, ReAct, Self-RAG or CRAG...", submit_mode="disable")

if not st.session_state.messages:
    selected = st.pills("Try asking", list(SUGGESTIONS), label_visibility="collapsed")
    if selected and not prompt:
        prompt = SUGGESTIONS[selected]

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    graph = load_graph()

    with st.chat_message("assistant"):
        steps, docs, answer, went_web = [], [], "No answer was generated.", False
        with st.status(":shimmer[Searching the papers]", type="compact") as status:
            for update in graph.stream({"question": prompt}, stream_mode="updates"):
                node, out = next(iter(update.items()))
                out = out or {}
                if node == "retrieve":
                    retrieved = len(out["documents"])
                    step, detail = f"Retrieved {retrieved} chunks", ", ".join(doc_label(d) for d in out["documents"])
                elif node == "grade_documents":
                    step = f"Graded chunks: {len(out['documents'])} of {retrieved} relevant"
                    detail = ", ".join(doc_label(d) for d in out["documents"]) or "None were relevant."
                elif node == "transform_query":
                    step, detail = "Rewrote the question for web search", f"`{out['question']}`"
                elif node == "web_search":
                    went_web = True
                    if out:
                        step, detail = "Searched the web", f"{len(out['documents'])} documents in context"
                    else:
                        step, detail = "Web search failed", "Answering with what was found locally."
                else:
                    step, detail = "Generated the answer", "Using the graded chunks" + (" and web results." if went_web else ".")
                    answer = out.get("generation", answer)

                if "documents" in out and node != "retrieve":
                    docs = out["documents"]
                steps.append(f"{step}: {detail}")
                with st.status(step, type="step", state="complete"):
                    st.caption(detail)

            steps_label = "Searched the papers and the web" if went_web else "Searched the papers"
            status.update(label=steps_label, state="complete")

        sources = list(dict.fromkeys(format_source(d) for d in docs))
        msg = {"role": "assistant", "content": answer, "steps": steps, "steps_label": steps_label, "sources": sources}
        st.markdown(answer)
        if sources:
            with st.expander("Sources", icon=":material/menu_book:", key=f"sources_{len(st.session_state.messages)}"):
                for source in sources:
                    st.markdown(f"- {source}")

    st.session_state.messages.append(msg)
