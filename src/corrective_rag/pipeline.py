"""Corrective RAG (CRAG) pipeline: retrieve -> grade -> (rewrite -> web search) -> generate."""

import os
from pathlib import Path
from typing import Annotated, List, Literal

from ddgs import DDGS
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.getenv("DATA_DIR", PROJECT_ROOT))
INDEX_DIR = Path(os.getenv("INDEX_DIR", PROJECT_ROOT / "faiss_index"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2")
GROQ_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
TOP_K = int(os.getenv("TOP_K", "4"))


# ---------------------------------------------------------------- vector store

def build_vector_store() -> FAISS:
    """Load the FAISS index from disk, or build it from the PDFs in DATA_DIR."""
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    if (INDEX_DIR / "index.faiss").exists():
        return FAISS.load_local(str(INDEX_DIR), embeddings, allow_dangerous_deserialization=True)

    pdfs = sorted(DATA_DIR.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No PDF files found in {DATA_DIR}")

    whole_pdfs = []
    for pdf in pdfs:
        whole_pdfs.extend(PyPDFLoader(str(pdf)).load())

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(whole_pdfs)
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(str(INDEX_DIR))
    return vector_store


# ---------------------------------------------------------------------- state

class GraphState(BaseModel):
    question: str
    messages: Annotated[List[AnyMessage], add_messages] = Field(default_factory=list)
    documents: List[Document] = Field(default_factory=list)
    feedback: str = ""
    generation: str = ""


# -------------------------------------------------------------------- prompts

GRADER_PROMPT = SystemMessage(content="""You are a relevance grader in a Corrective RAG pipeline.
You will receive a user QUESTION and one retrieved DOCUMENT chunk from a research paper.

Decide whether the document contains information that helps answer the question.

Grade "Yes" if the chunk:
- directly answers the question, or part of it, or
- contains facts, definitions, methods, or results that are clearly about the question's topic.

Grade "No" if the chunk:
- is only loosely related (shares a keyword but not the meaning),
- is a reference list, author list, acknowledgements, table of contents, or other boilerplate,
- is about a different concept than the one asked about.

Be strict: the goal is to filter out irrelevant chunks, not to answer the question.
Judge only the given chunk; do not use outside knowledge to fill gaps.""")

REWRITE_PROMPT = SystemMessage(content="""You are a query rewriter in a Corrective RAG pipeline.
The user's QUESTION could not be answered well from the local document store, so it will be sent to a web search engine.

Rewrite the question into a single, concise web search query that is most likely to return relevant, factual results.

Guidelines:
- Keep the original intent and all key entities, names, acronyms, and technical terms.
- Expand ambiguous acronyms when it helps (e.g. "RAG" -> "retrieval-augmented generation").
- Remove filler words, greetings, and conversational phrasing.
- Prefer keywords and specific terms over full sentences; about 5-15 words.
- Do not answer the question, add facts, or introduce assumptions not present in it.
- Output only the rewritten query, with no explanation or quotes.""")

GENERATION_PROMPT = SystemMessage(content="""You are a research assistant answering questions about machine learning papers.
You will receive a user QUESTION and a set of CONTEXT passages. Each passage is either a chunk from a research paper or a web search result, and each one is labeled with its source.

Answer the question using only the information in the context.

Guidelines:
- Base every claim on the context; do not add facts from your own knowledge.
- If the context only partly answers the question, answer that part and clearly say what is missing.
- If the context does not contain the answer, say "I could not find this in the provided sources." and do not guess.
- Cite the source of each key claim in square brackets, e.g. [1706.03762v7.pdf, p. 3] or [https://example.com].
- If sources disagree, point out the disagreement instead of picking one silently.
- Be clear and concise: start with a direct answer, then add supporting detail. Use short paragraphs or bullet points when that is easier to read.
- Explain technical terms briefly when they are central to the answer.""")


class GradeDocument(BaseModel):
    """Relevance grade for one retrieved chunk."""
    relevant: Literal["Yes", "No"] = Field(description="'Yes' if the chunk is relevant to the question, else 'No'")


class RewrittenQuery(BaseModel):
    """Web-search-optimized version of the user's question."""
    query: str = Field(description="The rewritten web search query")


def doc_label(doc: Document) -> str:
    """Short citation label: 'file.pdf, p. N' for PDF chunks, the URL for web results."""
    source = os.path.basename(doc.metadata.get("source", "unknown"))
    page = doc.metadata.get("page")
    return f"{source}, p. {page + 1}" if page is not None else doc.metadata.get("source", "unknown")


def format_docs(docs: List[Document]) -> str:
    return "\n\n---\n\n".join(f"[{doc_label(doc)}]\n{doc.page_content}" for doc in docs)


# ---------------------------------------------------------------------- graph

def build_graph(vector_store: FAISS):
    retriever = vector_store.as_retriever(search_kwargs={"k": TOP_K})
    llm = ChatGroq(model=GROQ_MODEL, temperature=0)
    grader = llm.with_structured_output(GradeDocument)
    rewriter = llm.with_structured_output(RewrittenQuery)

    def retrieve_node(state: GraphState) -> dict:
        return {"documents": retriever.invoke(state.question)}

    def grade_document(state: GraphState) -> dict:
        relevant_docs = []
        for doc in state.documents:
            result = grader.invoke([
                GRADER_PROMPT,
                HumanMessage(content=f"QUESTION:\n{state.question}\n\nDOCUMENT:\n{doc.page_content}"),
            ])
            if result.relevant == "Yes":
                relevant_docs.append(doc)
        return {"documents": relevant_docs, "feedback": "Yes" if relevant_docs else "No"}

    def transform_query_node(state: GraphState) -> dict:
        result = rewriter.invoke([REWRITE_PROMPT, HumanMessage(content=f"QUESTION:\n{state.question}")])
        return {"question": result.query}

    def web_search_node(state: GraphState) -> dict:
        try:
            results = DDGS().text(state.question, max_results=5)
        except Exception as e:
            print(f"Web search failed: {e}")
            return {}
        web_docs = [
            Document(page_content=r["body"], metadata={"source": r["href"], "title": r["title"]})
            for r in results
        ]
        return {"documents": state.documents + web_docs}

    def generate_node(state: GraphState) -> dict:
        response = llm.invoke([
            GENERATION_PROMPT,
            HumanMessage(content=f"QUESTION:\n{state.question}\n\nCONTEXT:\n{format_docs(state.documents)}"),
        ])
        return {"generation": response.content, "messages": [response]}

    def router_node(state: GraphState) -> Literal["transform_query", "generate"]:
        return "generate" if state.feedback == "Yes" else "transform_query"

    graph = StateGraph(GraphState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("grade_documents", grade_document)
    graph.add_node("transform_query", transform_query_node)
    graph.add_node("web_search", web_search_node)
    graph.add_node("generate", generate_node)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "grade_documents")
    graph.add_conditional_edges("grade_documents", router_node)
    graph.add_edge("transform_query", "web_search")
    graph.add_edge("web_search", "generate")
    graph.add_edge("generate", END)

    return graph.compile()
