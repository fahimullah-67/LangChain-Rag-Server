from rag.chain import build_chain

chain = build_chain()

def ask_question(question: str):
    return chain.invoke(question)