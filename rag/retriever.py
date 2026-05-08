from db.mongo import get_vector_collection
from langchain.schema import Document

def load_documents():
    collection = get_vector_collection()
    docs = list(collection.find({}))

    documents = []

    for doc in docs:
        for chunk in doc.get("textChunks", []):
            documents.append(
                Document(
                    page_content=chunk,
                    splitter = RecursiveCharacterTextSplitter(
                            chunk_size=400,
                            chunk_overlap=80
                        )
                )
            )

    return documents