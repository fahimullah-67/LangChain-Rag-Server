import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from rag.embeddings import get_embeddings
from rag.prompt import get_prompt

load_dotenv()

CHROMA_DIR = "./chroma_db"

llm = ChatGoogleGenerativeAI(
    model="gemini-flash-latest",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.2
)

embeddings = get_embeddings()
vector_db = Chroma(
    persist_directory=CHROMA_DIR,
    embedding_function=embeddings
)

def analyze_query(query):
    query = query.lower()
    if "car" in query:
        return {"typeLoan": "auto"}
    if "home" in query:
        return {"typeLoan": "home"}
    if "personal" in query:
        return {"typeLoan": "personal"}
    return {}

def format_history(history):
    if not history:
        return "No previous conversation."
    formatted = []
    for msg in history[-3:]:
        role = "User" if msg["role"] == "user" else "Assistant"
        formatted.append(f"{role}: {msg['content']}")
    return "\n".join(formatted)

def format_docs_with_metadata(docs):
    """Return both the formatted context string and a list of scheme IDs."""
    formatted = []
    scheme_ids = []
    for d in docs:
        text = d.page_content.strip()
        meta = d.metadata
        # Extract scheme ID (priority: _id, id, scheme_id)
        scheme_id = meta.get('_id', meta.get('id', meta.get('scheme_id', 'N/A')))
        if scheme_id != 'N/A':
            scheme_ids.append(scheme_id)

        # Build tenure string
        tenure_min = meta.get('tenureMin', 'N/A')
        tenure_max = meta.get('tenureMax', 'N/A')
        if tenure_min != 'N/A' and tenure_max != 'N/A':
            tenure = f"{tenure_min} to {tenure_max} months"
        else:
            tenure = meta.get('tenure', 'N/A')

        # Build eligibility string
        eligibility = meta.get('eligibility', meta.get('eligibilityCriteria', 'N/A'))
        # Build max amount
        max_amount = meta.get('max_amount', meta.get('maxAmount', 'N/A'))
        # Build fees
        fees = meta.get('fees', meta.get('processingFee', 'N/A'))

        scheme_info = f"""
Scheme ID: {scheme_id}
Bank: {meta.get('bank', 'N/A')}
Scheme: {meta.get('scheme_name', 'N/A')}
Type: {meta.get('typeLoan', meta.get('type', 'N/A'))}
Interest Rate: {meta.get('interestRate', 'N/A')}%
Tenure: {tenure}
Eligibility: {eligibility}
Max Amount: {max_amount}
Fees: {fees}
---
Details: {text}
"""
        formatted.append(scheme_info)
    return "\n\n".join(formatted), scheme_ids

def run_rag(query: str, history: list = None):
    # 1. Retrieve documents
    filters = analyze_query(query)
    retriever = vector_db.as_retriever(
        search_kwargs={
            "k": 5,
            "filter": filters if filters else None
        }
    )
    docs = retriever.invoke(query)
    context, scheme_ids = format_docs_with_metadata(docs)

    # 2. Prepare scheme ID info for the prompt
    scheme_id_text = ", ".join(scheme_ids) if scheme_ids else "None"

    # 3. Get prompt template
    prompt = get_prompt()

    # 4. Format history
    history_text = format_history(history) if history else "No previous conversation."

    # 5. Build chain
    chain = (
        {
            "context": lambda x: x["context"],
            "question": lambda x: x["question"],
            "history": lambda x: x["history"],
            "schemeId": lambda x: x["schemeId"],
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain.invoke({
        "context": context,
        "question": query,
        "history": history_text,
        "schemeId": scheme_id_text
    })

# Optional: test block (only runs if script is executed directly)
if __name__ == "__main__":
    docs = vector_db.similarity_search("HBL Islamic Home Finance", k=1)
    if docs:
        print("Metadata keys:", docs[0].metadata.keys())
        print("Full metadata:", docs[0].metadata)