import os

from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from ingest import (
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    create_embedding_model,
)


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

# load_dotenv()


# ============================================================
# CONFIGURATION
# ============================================================

LLM_MODEL = "qwen2.5:3b"

TOP_K = 4


# ============================================================
# LOAD CHROMADB
# ============================================================

def load_vector_database(embeddings):

    print("Loading ChromaDB...")

    vectorstore = Chroma(
        persist_directory=CHROMA_DIR,
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings
    )

    print("ChromaDB loaded successfully.")

    return vectorstore


# ============================================================
# CREATE LLM
# ============================================================

def create_llm():

    llm = ChatOllama(
        model=LLM_MODEL,
        temperature=0
    )

    print(
        f"LLM loaded: {LLM_MODEL}"
    )

    return llm


# ============================================================
# CREATE RAG PROMPT
# ============================================================

def create_prompt():

    return ChatPromptTemplate.from_template(
        """
You are a professional Health Insurance Policy Assistant.

Answer the user's question using ONLY the information
provided in the retrieved policy documents.

IMPORTANT RULES:

1. Do not use outside knowledge.

2. Do not invent information.

3. Do not guess policy benefits, coverage,
   waiting periods, exclusions, premiums,
   claim conditions, or other policy details.

4. If the answer is not available in the
   retrieved policy documents, respond with EXACTLY:

I could not find this information in the provided policy documents.

5. If information from multiple companies is relevant,
   clearly distinguish between them.

6. Keep the answer clear and easy to understand.

7. Do not mention retrieved chunks.

8. Do not mention the retrieval process.

9. Do not create source information yourself.

10. If the documents contain different conditions
    for different policies, do not combine them.

------------------------------------------------------------
RETRIEVED POLICY INFORMATION
------------------------------------------------------------

{context}

------------------------------------------------------------
USER QUESTION
------------------------------------------------------------

{question}

------------------------------------------------------------
ANSWER
------------------------------------------------------------
"""
    )


# ============================================================
# RETRIEVE DOCUMENTS
# ============================================================

def retrieve_documents(vectorstore, question):

    retriever = vectorstore.as_retriever(
        search_kwargs={
            "k": TOP_K
        }
    )

    documents = retriever.invoke(question)

    return documents


# ============================================================
# FORMAT CONTEXT
# ============================================================

def format_context(documents):

    context_parts = []

    for index, document in enumerate(
        documents,
        start=1
    ):

        metadata = document.metadata

        company = metadata.get(
            "company",
            "Unknown"
        )

        source_file = metadata.get(
            "source_file",
            "Unknown"
        )

        page = metadata.get(
            "page_number",
            metadata.get(
                "page",
                "Unknown"
            )
        )

        document_type = metadata.get(
            "document_type",
            "Unknown"
        )

        chunk_id = metadata.get(
            "chunk_id",
            "Unknown"
        )

        content = document.page_content

        context = f"""
SOURCE {index}

Company:
{company}

Document:
{document_type}

Source File:
{source_file}

Page:
{page}

Chunk ID:
{chunk_id}

Content:
{content}
"""

        context_parts.append(context)

    return "\n".join(context_parts)


# ============================================================
# CHECK IF ANSWER WAS NOT FOUND
# ============================================================

def answer_not_found(answer):

    normalized_answer = answer.strip().lower()

    not_found_message = (
        "i could not find this information "
        "in the provided policy documents."
    )

    return normalized_answer == not_found_message


# ============================================================
# DISPLAY SOURCES
# ============================================================

def display_sources(documents):

    print("\n")
    print("=" * 70)
    print("SOURCES")
    print("=" * 70)

    displayed_sources = set()

    source_number = 1

    for document in documents:

        metadata = document.metadata

        company = metadata.get(
            "company",
            "Unknown"
        )

        source_file = metadata.get(
            "source_file",
            "Unknown"
        )

        page = metadata.get(
            "page_number",
            metadata.get(
                "page",
                "Unknown"
            )
        )

        chunk_id = metadata.get(
            "chunk_id",
            "Unknown"
        )

        source_key = (
            company,
            source_file,
            page,
            chunk_id
        )

        if source_key in displayed_sources:
            continue

        displayed_sources.add(source_key)

        print(
            f"\n[{source_number}] {company}"
        )

        print(
            f"    Document : {source_file}"
        )

        print(
            f"    Page     : {page}"
        )

        print(
            f"    Chunk ID : {chunk_id}"
        )

        source_number += 1


# ============================================================
# GENERATE ANSWER
# ============================================================

def generate_answer(
    question,
    documents,
    llm,
    prompt
):

    context = format_context(documents)

    messages = prompt.format_messages(
        context=context,
        question=question
    )

    response = llm.invoke(messages)

    return response.content.strip()


# ============================================================
# PROCESS QUESTION
# ============================================================

def process_question(
    question,
    vectorstore,
    llm,
    prompt
):

    print("\nGenerating answer...")

    documents = retrieve_documents(
        vectorstore,
        question
    )

    if not documents:

        print(
            "\nI could not find this information "
            "in the provided policy documents."
        )

        return

    answer = generate_answer(
        question=question,
        documents=documents,
        llm=llm,
        prompt=prompt
    )

    print("\n")
    print("=" * 70)
    print("ANSWER")
    print("=" * 70)

    print(f"\n{answer}")

    # Show sources ONLY when an answer was found
    if not answer_not_found(answer):

        display_sources(documents)


# ============================================================
# CHAT LOOP
# ============================================================

def start_chat():

    print("\n")
    print("=" * 70)
    print("       HEALTH INSURANCE RAG CHATBOT")
    print("=" * 70)

    print("\nInitializing system...")

    # Load embedding model
    embeddings = create_embedding_model()

    # Load ChromaDB
    vectorstore = load_vector_database(
        embeddings
    )

    # Load LLM
    llm = create_llm()

    # Create prompt
    prompt = create_prompt()

    print("\n")
    print("=" * 70)
    print("CHATBOT READY")
    print("=" * 70)

    print(
        "\nAsk questions about your health insurance documents."
    )

    print(
        "Type 'exit' to close the chatbot."
    )

    # Continuous conversation
    while True:

        print("\n" + "-" * 70)

        question = input(
            "\nYou: "
        ).strip()

        # Empty input
        if not question:

            print(
                "Please enter a question."
            )

            continue

        # Exit
        if question.lower() in {
            "exit",
            "quit",
            "q"
        }:

            print(
                "\nThank you for using the "
                "Health Insurance RAG Chatbot."
            )

            break

        # Process question
        try:

            process_question(
                question=question,
                vectorstore=vectorstore,
                llm=llm,
                prompt=prompt
            )

        except Exception as error:

            print("\n")
            print("=" * 70)
            print("ERROR")
            print("=" * 70)

            print(
                f"\n{error}"
            )


# ============================================================
# MAIN
# ============================================================

def main():

    try:

        start_chat()

    except KeyboardInterrupt:

        print(
            "\n\nChatbot stopped."
        )

    except Exception as error:

        print("\n")
        print("=" * 70)
        print("APPLICATION ERROR")
        print("=" * 70)

        print(
            f"\n{error}"
        )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()