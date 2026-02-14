# src/main.py

from similarity_search import get_best_match
from slm_handler import generate_response
from rag_retriever import retrieve_rag_answer

# -------------------------
# Guardrails: sensitive/out-of-domain keywords
# -------------------------
SENSITIVE_KEYWORDS = [
    "password", "pin", "credit card number", "account number",
    "social security", "aadhaar", "personal info", "sensitive",
    "secret formula", "banking profits", "confidential"
]

def check_guardrails(query):
    """
    Returns True if query is safe, False if it contains sensitive/out-of-domain info.
    """
    query_lower = query.lower()
    for kw in SENSITIVE_KEYWORDS:
        if kw in query_lower:
            return False
    return True

# -------------------------
# Main query handler
# -------------------------
def handle_query(query):
    if not check_guardrails(query):
        return "Sorry, I cannot process sensitive or unsafe information."

    # Tier 1: Dataset similarity
    tier1_response = get_best_match(query)
    if tier1_response and tier1_response.strip():
        return tier1_response

    # Tier 2: RAG retrieval
    tier2_response = retrieve_rag_answer(query)
    if tier2_response and tier2_response.strip():
        return tier2_response

    # Tier 3: SLM fallback
    tier3_response = generate_response(query)
    if tier3_response and tier3_response.strip() and tier3_response != query:
        return tier3_response

    # Fallback
    return "I'm sorry, I do not have information on that topic."

# -------------------------
# CLI Loop
# -------------------------
if __name__ == "__main__":
    print("BFSI Call Center AI Assistant (type 'exit' to quit)")
    while True:
        user_query = input("\nEnter your query: ").strip()
        if user_query.lower() == "exit":
            break
        response = handle_query(user_query)
        print(f"\nResponse: {response}")
