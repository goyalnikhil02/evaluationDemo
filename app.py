import os
import asyncio
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv

# ADK and Toolbox Imports
from google import adk
from toolbox_core import ToolboxSyncClient
from google.adk.sessions import InMemorySessionService
from google.genai import types as genai_types

load_dotenv()
PROJECT_ID = os.getenv("PROJECT_ID", "your-project-id")
GOOGLE_CLOUD_LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
APP_NAME = "FroyoAgentOS"
USER = "default_user"
MODEL = os.getenv("MODEL", "gemini-2.5-flash")
TOOLBOX_URL = os.getenv("MCP_TOOLBOX_SERVER_URL", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

app = Flask(__name__)

all_tools = []
try:
    toolboxCore = ToolboxSyncClient(TOOLBOX_URL)
    
    # Strictly load the toolset mapped in your tools.yaml
    all_tools = toolboxCore.load_toolset("alloydb_tools")
    
    print(f"-> MCP Client: Successfully connected to {TOOLBOX_URL}")
    print(f"-> MCP Tools Loaded: {len(all_tools)} tools found.")
    for t in all_tools:
        # Safely extract the name or fall back to the string representation
        tool_name = getattr(t, 'name', getattr(t, 'tool_name', str(t)))
        print(f"   - {tool_name}")
except Exception as e:
    print(f"FATAL ERROR: Could not connect to MCP Toolbox Server at {TOOLBOX_URL}. Error: {e}")

store_manager_agent = adk.Agent(
    name="FroyoManager",
    model=MODEL,
    description="Store Manager Assistant for querying AlloyDB and BigQuery Froyo data.",
    tools=all_tools,
    instruction="""
    You are the Froyo Store Manager Assistant. You have access to real-time transactional data 
    (AlloyDB) and federated analytical data (BigQuery).
    
    OPERATING PROTOCOLS:
    1. If a customer asks about a product's allergens, use the 'check_allergens' tool. Remember to format the search term with wildcards (e.g., '%Midnight%').
    2. If a customer wants to place an order, use the 'place_order' tool.
    3. Always format your responses clearly. 
    4. Use markdown bullet points for lists. Do not output raw database IDs to the user.
    5. Be polite, concise, and helpful.
    """
)

# Using InMemorySessionService for local testing without external databases
session_service = InMemorySessionService()

runner = adk.Runner(
    agent=store_manager_agent,
    app_name=APP_NAME,
    session_service=session_service,
)

# Global session holder
global_session = None

async def initialize_session():
    global global_session
    try:
        global_session = await session_service.create_session(app_name=APP_NAME, user_id=USER)
        print(f"-> Session initialized successfully: {global_session.id}")
    except Exception as e:
        print(f"Error creating session: {e}")

# Create the session before handling requests
asyncio.run(initialize_session())

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
            print("\n=== STARTING AGENT RUN ===")
            async for event in runner.run_async(
                new_message=content,
                user_id=USER,
                session_id=global_session.id
            ):
                # 1. VERBOSE LOGGING: Print exactly what the ADK is doing
                print(f"[ADK EVENT TYPE]: {type(event).__name__}")
                
                # Check if it's a Tool Execution Event to see the DB results
                if hasattr(event, 'actions') and event.actions:
                     print(f"[ADK TOOL ACTION]: {event.actions}")
                if hasattr(event, 'output') and event.output:
                     print(f"[ADK TOOL OUTPUT]: {event.output}")

                # 2. Extract Text (covering all bases)
                if hasattr(event, 'text') and event.text:
                    accumulated_text.append(event.text)
                elif hasattr(event, 'content') and event.content:
                    if hasattr(event.content, 'parts') and event.content.parts:
                        for part in event.content.parts:
                            if hasattr(part, 'text') and part.text:
                                accumulated_text.append(part.text)
                elif hasattr(event, 'data') and hasattr(event.data, 'message') and event.data.message:
                    accumulated_text.append(str(event.data.message))

            print("=== AGENT RUN COMPLETE ===\n")
            final_reply = "".join(accumulated_text).strip()
            return final_reply
            
        except Exception as e:
            print(f"=== ADK RUNNER ERROR ===\n{e}\n===")
            import traceback
            traceback.print_exc()
            return f"Agent encountered an error: {str(e)}"

    # Run the async loop and format the response
    try:
        reply = asyncio.run(run_agent_loop())
        if not reply:
            reply = "I completed the request, but the text response was empty. Please check the backend console."
        return jsonify({"agent_reply": reply})
    except Exception as e:
        print(f"=== FLASK ROUTE ERROR ===\n{e}\n===")
        return jsonify({"agent_reply": "Internal server error. Please check backend logs."}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)
