import os
from google import genai
from google.genai import types
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Initialize the client (Uses your FREE API key from Google AI Studio)
client = genai.Client()
JUDGE_MODEL = "gemini-2.5-pro"

# ==========================================
# PHASE 1: EVALUATE TOOL ACCURACY
# ==========================================
tool_test = {
    "user_prompt": "Order 2 Midnight Swirls for Alice.",
    "agent_tool_call": '{"name": "place_order", "args": {"customer_name": "Alice", "product_name": "Midnight Swirl", "quantity": 2}}'
}

tool_judge_prompt = f"""
You are an impartial Agent Evaluator. Your job is to measure TOOL ACCURACY.
Look at the user's prompt and the JSON tool call the agent generated.
Did the agent select the 'place_order' tool and correctly extract the customer name (Alice), product (Midnight Swirl), and quantity (2)?
If yes, score 1. If no, score 0.

User Asked: {tool_test['user_prompt']}
Agent Tool Call: {tool_test['agent_tool_call']}

Provide your evaluation in this format:
SCORE: [0 or 1]
REASON: [Your explanation]
"""

print("=== PHASE 1: TOOL ACCURACY RESULT ===")
print(client.models.generate_content(model=JUDGE_MODEL, contents=tool_judge_prompt).text)


# ==========================================
# PHASE 2: EVALUATE GROUNDEDNESS
# ==========================================
groundedness_test = {
    "user_prompt": "What allergens are in the Volcanic Guava Burst?",
    "database_payload_returned": "{'allergen_name': 'None'}",
    "agent_final_response": "The Volcanic Guava Burst is completely allergen-free!"
}

groundedness_judge_prompt = f"""
You are an impartial Agent Evaluator. Your job is to measure GROUNDEDNESS.
Compare the Agent's Final Response to the Raw Database Payload. 
If the Agent included ANY information not found in the Database Payload, score it a 0.
If the Agent accurately represented the payload, score it a 1.

User Asked: {groundedness_test['user_prompt']}
Database Payload: {groundedness_test['database_payload_returned']}
Agent Response: {groundedness_test['agent_final_response']}

Provide your evaluation in this format:
SCORE: [0 or 1]
REASON: [Your explanation]
"""

print("\n=== PHASE 2: GROUNDEDNESS RESULT ===")
print(client.models.generate_content(model=JUDGE_MODEL, contents=groundedness_judge_prompt).text)
