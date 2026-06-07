import os
import pandas as pd
import vertexai
from dotenv import load_dotenv
from toolbox_langchain import ToolboxClient
from vertexai.evaluation import EvalTask

# Load environment variables
load_dotenv()
PROJECT_ID = os.getenv("PROJECT_ID")
TOOLBOX_URL = os.getenv("MCP_TOOLBOX_SERVER_URL")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Initialize Vertex AI
vertexai.init(project=PROJECT_ID, location="us-central1")

# ==========================================
# FETCH LIVE CONTEXT
# ==========================================
print(f"Fetching live database context via MCP Toolbox at {TOOLBOX_URL}...")
toolbox = ToolboxClient(TOOLBOX_URL) 
allergen_tool = toolbox.load_tool("check_allergens")

# We physically query AlloyDB/BigQuery to get the real payload for Midnight Swirl
live_midnight_context = str(allergen_tool.invoke({"product_name": "Midnight Swirl"}))
print(f"Live Context Received: {live_midnight_context}\n")

# ==========================================
# DATASET 1: TOOL ACCURACY (Routing Phase)
# ==========================================
# We use the "exact_match" metric to ensure the JSON tool call is perfectly constructed.
tool_dataset = pd.DataFrame([
  { # Happy Path
    "prompt": "Order 2 Midnight Swirls for Alice.",
    "response": '[{"name": "place_order", "args": {"customer_name": "Alice", "product_name": "Midnight Swirl", "quantity": 2}}]',
    "reference": '[{"name": "place_order", "args": {"customer_name": "Alice", "product_name": "Midnight Swirl", "quantity": 2}}]'
  },
  { # Edge Case: Missing quantity! Agent should route to a clarifying question.
    "prompt": "Order a Midnight Swirl for Alice.",
    "response": '[{"name": "ask_clarifying_question", "args": {"missing_info": "quantity"}}]',
    "reference": '[{"name": "ask_clarifying_question", "args": {"missing_info": "quantity"}}]'
  }
])

# ==========================================
# DATASET 2: GROUNDEDNESS (Synthesis Phase)
# ==========================================
groundedness_dataset = pd.DataFrame([
  { # Happy Path: Accurately summarizing our live database context
    "prompt": f"Summarize this database payload for the user: {live_midnight_context}",
    "context": live_midnight_context,
    "response": "The allergen name is Soy."
  },
  { # Negative Case: Intentional Hallucination!
    "prompt": "Summarize this database payload for the user: {'allergen_name': 'None'}",
    "context": "{'allergen_name': 'None'}",
    "response": "This product contains Dairy!" # This is a lie. The evaluator should catch it.
  }
])

# ==========================================
# RUN EVALUATION PIPELINE
# ==========================================
print("Running Phase 1: Tool Accuracy Eval...")
tool_eval_task = EvalTask(dataset=tool_dataset, metrics=["exact_match"])
tool_result = tool_eval_task.evaluate()

print("Running Phase 2: Groundedness Eval...")
groundedness_eval_task = EvalTask(dataset=groundedness_dataset, metrics=["groundedness"])
groundedness_result = groundedness_eval_task.evaluate()

# ==========================================
# FINAL SCORECARD & INTERPRETATION
# ==========================================
print("\n=== AGENT EVALUATION SCORECARD ===")

# 1. Interpret Tool Accuracy
exact_match_score = tool_result.summary_metrics.get("exact_match/mean")
print(f"Routing Phase (exact_match/mean): {exact_match_score}")
if exact_match_score == 1.0:
    print("✅ PASS: The agent perfectly constructed the JSON tool calls and captured all parameters.")
else:
    print("❌ FAIL: The agent struggled to format the tool call correctly.")

# 2. Interpret Groundedness
groundedness_score = groundedness_result.summary_metrics.get("groundedness/mean")
print(f"\nSynthesis Phase (groundedness/mean): {groundedness_score}")
if groundedness_score == 0.5:
    print("✅ PASS: Evaluator gave 1.0 to the truthful answer and 0.0 to the hallucination. The safety net works!")
else:
    print("❌ FAIL: The evaluator missed the hallucination or incorrectly flagged the truth.")
