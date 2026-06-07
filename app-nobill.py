# This file is created for readers who want to experience the ADK & orchestration steps of the app without setting up data in BigQuery or AlloyDB

import os
import asyncio
import pandas as pd
import uuid
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv

# ADK Imports (Notice we don't need toolbox_core here)
from google import adk
from google.adk.sessions import InMemorySessionService
from google.genai import types as genai_types

load_dotenv()
PROJECT_ID = os.getenv("PROJECT_ID", "your-project-id")
GOOGLE_CLOUD_LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
APP_NAME = "FroyoAgentOS"
USER = "default_user"
MODEL = os.getenv("MODEL", "gemini-2.5-flash")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

app = Flask(__name__)

# ==========================================
# 1. LOCAL DATA SETUP (Pandas instead of AlloyDB)
# ==========================================
# Try to load from CSV, fallback to in-memory DataFrame if not found
try:
    allergen_df = pd.read_csv('froyo_data.allergen.csv')
    consistsof_df = pd.read_csv('froyo_data.consistsof.csv')
    containsallergen_df = pd.read_csv('froyo_data.containsallergen.csv')
    froyo_data_materialized_df = pd.read_csv('froyo_data.froyo_data_materialized.csv')
    ingredient_df = pd.read_csv('froyo_data.ingredient.csv')
    product_df = pd.read_csv('froyo_data.product.csv')
    suppliedby_df = pd.read_csv('froyo_data.suppliedby.csv')
    supplier_df = pd.read_csv('froyo_data.supplier.csv')
    print("-> Local Data: Loaded all 8 CSV files successfully.")
except FileNotFoundError:
    print("-> Local Data: CSV files not found. Using fallback mock data.")
    product_df = pd.DataFrame({'product_id': [1], 'product_name': ['Midnight Swirl']})
    consistsof_df = pd.DataFrame({'product_id': [1], 'ingredient_id': ['ing_1']})
    ingredient_df = pd.DataFrame({'ingredient_id': ['i1'], 'ingredient_name': ['ing_1']})
    containsallergen_df = pd.DataFrame({'ingredient_id': ['i1'], 'allergen_name': ['Soy']})
    allergen_df = pd.DataFrame()
    froyo_data_materialized_df = pd.DataFrame()
    suppliedby_df = pd.DataFrame()
    supplier_df = pd.DataFrame()

# ==========================================
# 2. PYTHON NATIVE TOOLS (Instead of Toolbox SQL)
# ==========================================
def check_allergens(product_name: str) -> str:
    """Queries the local CSV data to find allergens for a product.
    
    Args:
        product_name: The name of the product to check (e.g. Midnight Swirl)
    """
    try:
        # 1. WHERE UPPER(p.product_name) LIKE UPPER($1)
        p = product_df[product_df['product_name'].str.contains(product_name, case=False, na=False)]
        if p.empty:
            return f"Product '{product_name}' not found in the catalog."
        
        # 2. INNER JOIN consistsof c ON p.product_id = c.product_id
        p_c = pd.merge(p, consistsof_df, on='product_id', how='inner')
        
        # 3. INNER JOIN ingredient i ON c.ingredient_id = i.ingredient_name
        # Note: consistsof has 'ingredient_id', ingredient has 'ingredient_name'
        p_c_i = pd.merge(p_c, ingredient_df, left_on='ingredient_id', right_on='ingredient_name', how='inner')
        
        # 4. INNER JOIN containsallergen a ON i.ingredient_id = a.ingredient_id
        # Because both consistsof and ingredient tables have an 'ingredient_id' column, 
        # Pandas renames them to _x and _y. We need the one from the ingredient table (right side).
        target_col = 'ingredient_id_y' if 'ingredient_id_y' in p_c_i.columns else 'ingredient_id'
        final_df = pd.merge(p_c_i, containsallergen_df, left_on=target_col, right_on='ingredient_id', how='inner')
        
        if final_df.empty:
            return f"{product_name} has no known allergens."
            
        allergens = final_df['allergen_name'].dropna().unique()
        allergen_str = ", ".join(allergens)
        
        return f"The allergens for {product_name} are: {allergen_str}"
    except Exception as e:
        return f"Error analyzing allergens: {str(e)}"

def place_order(customer_name: str, product_name: str, quantity: int) -> str:
    """Inserts a new live transaction. (Mocked for local CSV version)
    
    Args:
        customer_name: The name of the customer placing the order.
        product_name: The name of the product being ordered.
        quantity: The quantity of the product being ordered.
    """
    matches = product_df[product_df['product_name'].str.contains(product_name, case=False, na=False)]
    if matches.empty:
        return f"Cannot place order. Product '{product_name}' not found."
        
    real_product_name = matches.iloc[0]['product_name']
    product_id = matches.iloc[0]['product_id']
    
    order_id = str(uuid.uuid4())[:8] # Generate a fake order ID
    return f"SUCCESS: Order ID {order_id} placed for {customer_name} (Item: {quantity}x {real_product_name}, ID: {product_id})."

# Package our python functions as tools for the Agent
all_tools = [check_allergens, place_order]

# ==========================================
# 3. AGENT SETUP (Identical to production!)
# ==========================================
store_manager_agent = adk.Agent(
    name="FroyoManager",
    model=MODEL,
    description="Store Manager Assistant for querying Froyo data.",
    tools=all_tools,
    instruction="""
    You are the Froyo Store Manager Assistant. You have access to the local product catalog.
    
    OPERATING PROTOCOLS:
    1. If a customer asks about a product's allergens, use the 'check_allergens' tool. 
    2. If a customer wants to place an order, use the 'place_order' tool.
    3. Always format your responses clearly. 
    4. Use markdown bullet points for lists.
    5. Be polite, concise, and helpful.
    """
)

session_service = InMemorySessionService()

runner = adk.Runner(
    agent=store_manager_agent,
    app_name=APP_NAME,
    session_service=session_service,
)

global_session = None

async def initialize_session():
    global global_session
    try:
        global_session = await session_service.create_session(app_name=APP_NAME, user_id=USER)
        print(f"-> Session initialized successfully: {global_session.id}")
    except Exception as e:
        print(f"Error creating session: {e}")

asyncio.run(initialize_session())

# ==========================================
# 4. FLASK ROUTES (Identical to production!)
# ==========================================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    global global_session
    user_input = request.json.get('message', '')
    
    if not global_session:
        return jsonify({"agent_reply": "System is still initializing..."})

    content = genai_types.Content(role='user', parts=[genai_types.Part(text=user_input)])

    async def run_agent_loop():
        accumulated_text = []
        try:
            async for event in runner.run_async(
                new_message=content,
                user_id=USER,
                session_id=global_session.id
            ):
                if hasattr(event, 'text') and event.text:
                    accumulated_text.append(event.text)
                elif hasattr(event, 'content') and event.content:
                    if hasattr(event.content, 'parts') and event.content.parts:
                        for part in event.content.parts:
                            if hasattr(part, 'text') and part.text:
                                accumulated_text.append(part.text)
                elif hasattr(event, 'data') and hasattr(event.data, 'message') and event.data.message:
                    accumulated_text.append(str(event.data.message))

            return "".join(accumulated_text).strip()
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"Agent encountered an error: {str(e)}"

    try:
        reply = asyncio.run(run_agent_loop())
        if not reply:
            reply = "I completed the request, but the text response was empty."
        return jsonify({"agent_reply": reply})
    except Exception as e:
        return jsonify({"agent_reply": "Internal server error."}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)
