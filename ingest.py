from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = Path("data")

CHROMA_DIR = "chroma_db"

COLLECTION_NAME = "health_insurance_documents"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# ============================================================
# STEP 1 — GET COMPANY NAME
# ============================================================

def get_company_name(pdf_file):
    """
    Determine the insurance company from the PDF filename.
    """

    filename = pdf_file.stem.lower()

    if "hdfc" in filename and "ergo" in filename:
        return "HDFC ERGO"

    if "niva" in filename and "bupa" in filename:
        return "Niva Bupa"

    # Fallback
    return pdf_file.stem.replace("_", " ").title()


# ============================================================
# STEP 2 — LOAD PDF DOCUMENTS
# ============================================================

def load_documents():
    """
    Load all PDF files from the data directory.

    Adds useful metadata to every PDF page:
        - company
        - source_file
        - page
        - document_type
    """

    documents = []

    pdf_files = sorted(DATA_DIR.glob("*.pdf"))

    if not pdf_files:

        print("\nNo PDF files found in the data folder.")

        return documents

    print("\n" + "=" * 70)
    print("LOADING PDF DOCUMENTS")
    print("=" * 70)

    for pdf_file in pdf_files:

        print(f"\nLoading: {pdf_file.name}")

        try:

            loader = PyPDFLoader(str(pdf_file))

            pdf_documents = loader.load()

            company_name = get_company_name(
                pdf_file
            )

            for document in pdf_documents:

                # ------------------------------------------------
                # Existing metadata from PyPDFLoader
                # ------------------------------------------------

                page_number = document.metadata.get(
                    "page",
                    None
                )

                # Convert zero-based page number
                # to human-readable page number.
                if page_number is not None:
                    display_page = page_number + 1
                else:
                    display_page = "Unknown"

                # ------------------------------------------------
                # Add our own metadata
                # ------------------------------------------------

                document.metadata["company"] = company_name

                document.metadata["source_file"] = (
                    pdf_file.name
                )

                document.metadata["document_type"] = (
                    "Policy Document"
                )

                document.metadata["page_number"] = (
                    display_page
                )

            documents.extend(pdf_documents)

            print(
                f"   Company      : {company_name}"
            )

            print(
                f"   Pages loaded : {len(pdf_documents)}"
            )

        except Exception as error:

            print(
                f"\nError loading {pdf_file.name}"
            )

            print(
                f"Error: {error}"
            )

    return documents


# ============================================================
# STEP 3 — SPLIT DOCUMENTS INTO CHUNKS
# ============================================================

def split_documents(documents):
    """
    Split PDF pages into smaller chunks.

    Each chunk automatically keeps the metadata
    from its original document/page.
    """

    print("\n" + "=" * 70)
    print("SPLITTING DOCUMENTS INTO CHUNKS")
    print("=" * 70)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        length_function=len,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ]
    )

    chunks = text_splitter.split_documents(
        documents
    )

    print(
        f"\nTotal chunks created: {len(chunks)}"
    )

    return chunks


# ============================================================
# STEP 4 — ADD CHUNK METADATA
# ============================================================

def add_chunk_metadata(chunks):
    """
    Add a unique chunk ID to every chunk.

    Example:
        HDFC_ERGO_00001
        HDFC_ERGO_00002
        NIVA_BUPA_00001
    """

    print("\n" + "=" * 70)
    print("ADDING CHUNK METADATA")
    print("=" * 70)

    company_counters = {}

    for chunk in chunks:

        company = chunk.metadata.get(
            "company",
            "Unknown"
        )

        # Create a safe prefix for the chunk ID
        company_prefix = (
            company
            .upper()
            .replace(" ", "_")
            .replace("-", "_")
        )

        # Initialize counter
        if company_prefix not in company_counters:

            company_counters[company_prefix] = 0

        company_counters[company_prefix] += 1

        chunk_number = company_counters[
            company_prefix
        ]

        # Create chunk ID
        chunk_id = (
            f"{company_prefix}_"
            f"{chunk_number:05d}"
        )

        chunk.metadata["chunk_id"] = chunk_id

    print(
        f"\nChunk metadata added successfully."
    )

    return chunks


# ============================================================
# STEP 5 — DISPLAY SAMPLE CHUNKS
# ============================================================

def display_sample_chunks(chunks, number_of_samples=5):
    """
    Display a few chunks so you can verify
    that the metadata looks correct.
    """

    print("\n" + "=" * 70)
    print("SAMPLE CHUNKS")
    print("=" * 70)

    samples = chunks[
        :number_of_samples
    ]

    for index, chunk in enumerate(
        samples,
        start=1
    ):

        metadata = chunk.metadata

        print("\n" + "-" * 70)

        print(
            f"SAMPLE CHUNK {index}"
        )

        print("-" * 70)

        print(
            f"Company      : "
            f"{metadata.get('company', 'Unknown')}"
        )

        print(
            f"Source file  : "
            f"{metadata.get('source_file', 'Unknown')}"
        )

        print(
            f"Document     : "
            f"{metadata.get('document_type', 'Unknown')}"
        )

        print(
            f"Page         : "
            f"{metadata.get('page_number', 'Unknown')}"
        )

        print(
            f"Chunk ID     : "
            f"{metadata.get('chunk_id', 'Unknown')}"
        )

        print("\nContent preview:")

        content = chunk.page_content

        if len(content) > 500:

            content = content[:500] + "..."

        print(content)


# ============================================================
# STEP 6 — CREATE EMBEDDING MODEL
# ============================================================

def create_embedding_model():
    """
    Create the Hugging Face embedding model.
    """

    print("\n" + "=" * 70)
    print("CREATING EMBEDDING MODEL")
    print("=" * 70)

    print(
        f"\nModel: {EMBEDDING_MODEL}"
    )

    print(
        "Device: CPU"
    )

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={
            "device": "cpu"
        },
        encode_kwargs={
            "normalize_embeddings": True
        }
    )

    print(
        "\nEmbedding model loaded successfully."
    )

    return embeddings


# ============================================================
# STEP 7 — CREATE CHROMADB
# ============================================================

def create_vector_database(
    chunks,
    embeddings
):
    """
    Convert document chunks into embeddings
    and store them in ChromaDB.
    """

    print("\n" + "=" * 70)
    print("CREATING CHROMADB VECTOR DATABASE")
    print("=" * 70)

    print(
        "\nCreating embeddings for document chunks..."
    )

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
        collection_name=COLLECTION_NAME
    )

    print(
        "\nChromaDB created successfully."
    )

    return vectorstore


# ============================================================
# STEP 8 — MAIN FUNCTION
# ============================================================

def main():

    print("\n")

    print("=" * 70)
    print(
        "       HEALTH INSURANCE RAG"
    )
    print(
        "       DOCUMENT INGESTION"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # Step 1 — Load PDFs
    # --------------------------------------------------------

    documents = load_documents()

    if not documents:

        print(
            "\nNo documents were loaded."
        )

        return

    print("\n" + "-" * 70)

    print(
        f"Total pages loaded: "
        f"{len(documents)}"
    )

    print("-" * 70)

    # --------------------------------------------------------
    # Step 2 — Split into chunks
    # --------------------------------------------------------

    chunks = split_documents(
        documents
    )

    if not chunks:

        print(
            "\nNo chunks were created."
        )

        return

    # --------------------------------------------------------
    # Step 3 — Add metadata
    # --------------------------------------------------------

    chunks = add_chunk_metadata(
        chunks
    )

    # --------------------------------------------------------
    # Step 4 — Display sample chunks
    # --------------------------------------------------------

    display_sample_chunks(
        chunks,
        number_of_samples=5
    )

    # --------------------------------------------------------
    # Step 5 — Create embedding model
    # --------------------------------------------------------

    embeddings = create_embedding_model()

    # --------------------------------------------------------
    # Step 6 — Create ChromaDB
    # --------------------------------------------------------

    vectorstore = create_vector_database(
        chunks,
        embeddings
    )

    # --------------------------------------------------------
    # Final information
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print(
        "       INGESTION COMPLETED SUCCESSFULLY"
    )
    print("=" * 70)

    print(
        "\nPDF documents loaded       : "
        f"{len(documents)}"
    )

    print(
        "Total chunks created       : "
        f"{len(chunks)}"
    )

    print(
        "Embedding model            : "
        f"{EMBEDDING_MODEL}"
    )

    print(
        "ChromaDB directory         : "
        f"{CHROMA_DIR}"
    )

    print(
        "Collection                 : "
        f"{COLLECTION_NAME}"
    )

    print("\nMetadata stored for chunks:")

    print("  • company")
    print("  • source_file")
    print("  • document_type")
    print("  • page_number")
    print("  • chunk_id")

    print("\n" + "=" * 70)

    print(
        "You can now test document retrieval."
    )

    print("=" * 70)


# ============================================================
# RUN PROGRAM
# ============================================================

if __name__ == "__main__":
    main()