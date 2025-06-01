from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langgraph.prebuilt import create_react_agent
import os
from dotenv import load_dotenv
from datetime import datetime
from langchain_core.tools import tool
import redis
from langchain_unstructured import UnstructuredLoader
from langchain_community.document_loaders import JSONLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_groq import ChatGroq
from langgraph.checkpoint.redis import  RedisSaver
from langchain_core.rate_limiters import InMemoryRateLimiter
from langgraph_supervisor import create_supervisor
rate_limiter = InMemoryRateLimiter(
    requests_per_second=0.5,          # 30 requests per minute
    check_every_n_seconds=0.2,        # Check 5 times per second
    max_bucket_size=10                # Allows some burst capacity
)
load_dotenv()
nvidia_api_key = os.getenv("NVIDIA_API_KEY")
nvidia_model= ChatNVIDIA(
    model="meta/llama-3.3-70b-instruct",
    nvidia_api_key=nvidia_api_key,
    temperature=0.2,
)
groq_model = ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    api_key=os.getenv("GROQ_API_KEY"),
    rate_limiter=rate_limiter,
)

# Initialize shared memory using redis-py
# redis_client = redis.StrictRedis(host="localhost", port=6379, db=0)
# @tool
# def store_metadata(format, intent):
#     """stores  format, and intent metadata in shared memory."""
#     metadata = {
#         "timestamp": datetime.now().isoformat(),
#         "classification": {
#             "format": format,
#             "intent": intent
#         }
#     }
#     redis_client.set(f"metadata:{datetime.now().timestamp()}", str(metadata))
#     return f"Metadata stored: {metadata}"
@tool
def classify_input(file_path: str):
    """Classifies the input file path and returns its data"""
    if file_path.endswith('.pdf'):
        loader = PyPDFLoader(file_path)
        data = loader.load()
        return data
    elif file_path.endswith('.json'):
        loader = JSONLoader(file_path, jq_schema=".[]")
        data = loader.load()
        return data
    elif file_path.endswith('.eml'):
        loader = UnstructuredLoader(file_path)
        data = loader.load()
        return data
# Enhance classifier agent with few-shot examples and schema matching
@tool
def escalate():
    """Escalates the issue to a human agent."""
    return "The issue has been escalated to a human agent for further assistance."
def log():
    """Logs the classification results."""
    return "Classification results have been logged successfully."
REDIS_URI = "redis://localhost:6379"
memory=None
with RedisSaver.from_conn_string(REDIS_URI) as checkpointer:
    # Initialize Redis indices (only needed once)
    checkpointer.setup()
    memory=checkpointer
email_agent=create_react_agent(name="email_agent", model=groq_model,tools=[escalate,log], prompt="You are an email classification agent.use escalate and log tool based on intent")
classifier_agent = create_supervisor(
    model=groq_model,
    agents=[email_agent],
    tools=[classify_input],
    output_mode="full_history",
    prompt="""
    You are a supervisor that classifies user input format to Email, PDF, JSON and identifies intents such as RFQ, Complaint, Invoice, Regulation, Fraud Risk.
    1.use the tools provided to classify the input.
    2.route to email agent if the input is an email.
    3.output the content""",
    add_handoff_back_messages=True
     ).compile(checkpointer=memory)
    # Use the agent with a specific thread ID to maintain conversation state
config = {"configurable": {"thread_id": "user205"}}
def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()
inputs=  {"messages": [{"role": "user", "content": "classify and pass to email agent /home/prajith/Desktop/flowbit/sample_email.eml"}]}
print_stream(classifier_agent.stream(inputs, config=config, stream_mode="values"))



