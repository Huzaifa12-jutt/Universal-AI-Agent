# ==========================
# rag/loader.py
# ==========================

from langchain_community.document_loaders import PyPDFLoader


def load_pdf(pdf_path, source_name=None):
    """
    Load a single PDF and return all pages as LangChain Documents.
    Tags every page with a clean 'source' filename in metadata so
    answers can be traced back to the right file later.
    """

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    label = source_name or pdf_path

    for doc in documents:
        doc.metadata["source"] = label

    return documents


def load_multiple_pdfs(pdf_paths_with_names):
    """
    Load several PDFs at once.

    pdf_paths_with_names: list of (path, display_name) tuples.
    Returns a combined list of Documents and a per-file page count dict.
    """

    all_documents = []
    stats = {}

    for path, name in pdf_paths_with_names:
        docs = load_pdf(path, source_name=name)
        all_documents.extend(docs)
        stats[name] = len(docs)

    return all_documents, stats
