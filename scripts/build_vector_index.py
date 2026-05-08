import os
from pymongo import MongoClient
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = "clics"
COLLECTION = "loanschemes"          # Your collection name
CHROMA_DIR = "./chroma_db"

# 1. CONNECT DB
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COLLECTION]

# 2. FETCH DATA with bank lookup
data = list(collection.aggregate([
    {
        "$lookup": {
            "from": "banks",
            "localField": "bankId",
            "foreignField": "_id",
            "as": "bank"
        }
    },
    {
        "$unwind": {
            "path": "$bank",
            "preserveNullAndEmptyArrays": True
        }
    }
]))

print(f"📊 Found {len(data)} schemes")

documents = []

for item in data:
    # Convert ObjectId to string for metadata
    scheme_id = str(item["_id"])
    bank_name = item.get("bank", {}).get("name", "Unknown Bank")

    # Build rich page content (what the LLM will see)
    text = f"""
Bank: {bank_name}
Scheme: {item.get('schemeName', 'N/A')}
Type: {item.get('typeLoan', 'N/A')}
Interest Rate: {item.get('interestRate', 'N/A')}%
Tenure: {item.get('tenureMin', 'N/A')} to {item.get('tenureMax', 'N/A')} months
Processing Fee: {item.get('processingFee', 'N/A')}
Late Payment Fee: PKR {item.get('latePaymentFee', 'N/A')}
Minimum Salary: PKR {item.get('minSalaryRequired', 'N/A')}
Age Range: {item.get('ageMin', 'N/A')} – {item.get('ageMax', 'N/A')} years
Islamic: {'Yes' if item.get('isIslamic') else 'No'}
Status: {item.get('status', 'N/A')}

Eligibility Criteria:
{item.get('eligibilityCriteria', 'N/A')}

Required Documents:
{item.get('requiredDocuments', 'N/A')}

Description:
{item.get('description', 'N/A')}
"""
    # Remove excessive whitespace
    text = "\n".join(line.strip() for line in text.splitlines() if line.strip())

    # Metadata – includes ALL fields needed for retrieval and for generating links/details
    metadata = {
        "_id": scheme_id,                    # 🔥 CRITICAL for links
        "bank": bank_name,
        "scheme_name": item.get('schemeName', 'N/A'),
        "typeLoan": item.get('typeLoan', 'N/A'),
        "interestRate": item.get('interestRate', 0),
        "tenureMin": item.get('tenureMin', 'N/A'),
        "tenureMax": item.get('tenureMax', 'N/A'),
        "processingFee": item.get('processingFee', 'N/A'),
        "latePaymentFee": item.get('latePaymentFee', 'N/A'),
        "minSalaryRequired": item.get('minSalaryRequired', 'N/A'),
        "ageMin": item.get('ageMin', 'N/A'),
        "ageMax": item.get('ageMax', 'N/A'),
        "isIslamic": item.get('isIslamic', False),
        "status": item.get('status', 'N/A'),
        "eligibility": item.get('eligibilityCriteria', 'N/A'),
        "documents": item.get('requiredDocuments', 'N/A'),
        "description": item.get('description', 'N/A'),
    }

    documents.append(Document(page_content=text, metadata=metadata))

# 3. SPLIT TEXT (optional, but keep chunking for long descriptions)
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = splitter.split_documents(documents)
print(f"✂️  Created {len(chunks)} chunks")

# 4. EMBEDDINGS
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 5. CREATE NEW CHROMA DB (overwrite old one)
# Remove existing directory if you want a clean rebuild
import shutil
if os.path.exists(CHROMA_DIR):
    shutil.rmtree(CHROMA_DIR)
    print(f"🗑️  Removed old Chroma DB at {CHROMA_DIR}")

vector_db = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=CHROMA_DIR
)

print(f"✅ Vector DB created successfully at {CHROMA_DIR}")
print(f"   Stored {len(chunks)} chunks with full metadata including _id")


# Fetch MongoDB data
# Convert → chunks
# Convert → embeddings
# Store → Chroma