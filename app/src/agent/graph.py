from typing import Literal, List,Dict,Any
from langchain_groq import ChatGroq
from langgraph.types import Command
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_core.messages import AIMessage,HumanMessage
from pydantic import BaseModel, Field
import os
from enum import Enum
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict
from typing import Annotated
from langchain_unstructured import UnstructuredLoader
from langchain_community.document_loaders import JSONLoader
from langchain_community.document_loaders import PyPDFLoader
from langgraph.prebuilt import ToolNode
from langchain.schema import Document
from langgraph.checkpoint.redis import RedisSaver
import requests
model = ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    api_key=os.getenv("GROQ_API_KEY"),
    
)
class MessageClassifier(BaseModel):
    next_agent: Literal["email_agent", "pdf_agent", "json_agent", "chat_agent"] = Field(
        ...,
        description="The next agent to call based on the data"
    )
    file_path:str | None = Field(
        ...,
        description="file path"
    )
    data:str | None = Field(
        ...,
        description="data to be processed by the next agent"
    )
class EmailTone(str, Enum):
    ESCALATION = "escalation"
    POLITE = "polite"
    THREATENING = "threatening"
    NEUTRAL = "neutral"
    URGENT = "urgent"

class UrgencyLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class EmailAnalysis(BaseModel):
    sender: str = Field(description="The email sender's address")
    urgency: UrgencyLevel = Field(description="The urgency level of the email")
    issue: str = Field(description="The main issue or request in the email")
    tone: EmailTone = Field(description="The tone of the email")
    action_needed: Literal["escalate", "routine"] = Field(description="Action to take based on tone and urgency")
    summary: str = Field(description="Brief summary of the email content")

class JsonValidationResult(BaseModel):
    is_valid: bool = Field(description="Whether the JSON data conforms to expected schema")
    missing_fields: List[str] = Field(description="List of required fields that are missing")
    type_errors: List[str] = Field(description="List of fields with incorrect data types")
    anomalies: List[str] = Field(description="Other anomalies detected in the data")
    summary: str = Field(description="Summary of validation results")
    alert_needed: bool = Field(description="Whether an alert needs to be logged")

class PdfAnalysisResult(BaseModel):
    document_type: Literal["invoice", "policy", "other"] = Field(description="Type of PDF document")
    extracted_fields:dict = Field(description="Key fields extracted from the document")
    total_amount: float| None = Field(description="Total amount if document is an invoice")
    flagged_keywords: List[str] = Field(description="List of flagged keywords found in policy documents")
    needs_attention: bool = Field(description="Whether the document needs special attention")
    summary: str = Field(description="Summary of the document analysis")

class State(TypedDict):
    messages: Annotated[list, add_messages]
def classify_input(file_path: str):
    """Classifies the input file path and returns its data"""
    if file_path.endswith('.pdf'):
        loader = PyPDFLoader(file_path)
        data = loader.load()
        return data[0].page_content
    elif file_path.endswith('.json'):
        # Simplified JSON handling to avoid nested structure issues
        import json
        try:
            with open(file_path, 'r') as file:
                data = json.load(file)
            # Convert to flattened string representation
            json_content = "\n".join([f"{key}: {value}" for key, value in data.items()])
            return [Document(page_content=f"JSON Content:\n{json_content}")]
        except Exception as e:
            return [Document(page_content=f"Error loading JSON: {str(e)}")]
    elif file_path.endswith('.eml'):
        loader = UnstructuredLoader(file_path)
        data = loader.load()
        return data


def supervisor(state: State) -> Command[Literal["email_agent", "pdf_agent", "json_agent", "chat_agent"]]:
    last_message = state["messages"]
    supervisor_model = model.with_structured_output(MessageClassifier)
    
    # Enhanced system prompt for better classification
    system_prompt = """
    Classify the user input to determine the next agent:
    - If the input contains a file path ending in '.pdf', return 'pdf_agent'.
    - If the input contains a file path ending in '.json', return 'json_agent'.
    - If the input contains a file path ending in '.eml', return 'email_agent'.
    - If no file path is provided or the input is a general question, return 'chat_agent'.
    
    Examples:
    - Input: "File: invoice.pdf" → next_agent: pdf_agent, file_path: invoice.pdf, data: None
    - Input: "File: webhook.json" → next_agent: json_agent, file_path: webhook.json, data: None
    - Input: "File: complaint.eml" → next_agent: email_agent, file_path: complaint.eml, data: None
    - Input: "What is the weather today?" → next_agent: chat_agent, file_path: None, data: None
    - Input: "Can you help me?" → next_agent: chat_agent, file_path: None, data: None
    """
    
    response = supervisor_model.invoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": last_message[-1].content}
    ])
    # Load data if a file path is provided

    data = classify_input(response.file_path) if response.file_path!=None else None
    
    # Format the data as a string for inclusion in the message
    
    data_content = str(data) if data else "No file data provided"
    
    # Create a structured message for the state
    response_content = f"""
    Classified input:
    - Next agent: {response.next_agent}
    - File path: {response.file_path or 'None'}
    - Data: {data_content}
    """
    
    # Return a Command with a proper AIMessage
    return Command(
        goto=response.next_agent,
        update={"messages": [{"role": "assistant", "content": response_content}]}
    )
def simulate_crm_notification(email_data: EmailAnalysis):
    """
    Simulates notifying a CRM system about an escalated email.
    In a real implementation, this would make an API call to a CRM system.
    """
    print(f"[CRM NOTIFICATION] Escalating email from {email_data.sender}")
    print(f"[CRM NOTIFICATION] Urgency: {email_data.urgency.value}")
    print(f"[CRM NOTIFICATION] Issue: {email_data.issue}")
    print(f"[CRM NOTIFICATION] Tone detected: {email_data.tone.value}")
    
    # Simulate API response
    return {
        "status": "success",
        "ticket_id": "CRM-" + str(hash(email_data.sender + email_data.issue) % 10000),
        "timestamp": "2025-05-31T12:00:00Z"
    }

def log_routine_email(email_data: EmailAnalysis):
    """
    Logs a routine email that doesn't require escalation.
    """
    print(f"[EMAIL LOG] Routine email from {email_data.sender}")
    print(f"[EMAIL LOG] Urgency: {email_data.urgency.value}")
    print(f"[EMAIL LOG] Issue: {email_data.issue}")
    print(f"[EMAIL LOG] Action: Logged and closed")
    
    return {
        "status": "logged",
        "log_id": "LOG-" + str(hash(email_data.sender) % 10000),
        "timestamp": "2025-05-31T12:00:00Z"
    }

def email_agent(state: State):
    """
    Enhanced email agent that:
    1. Extracts structured fields (sender, urgency, issue/request)
    2. Identifies tone (escalation, polite, threatening)
    3. Triggers actions based on tone + urgency
    """
    last_message = state["messages"]
    
    # Set up structured output for email analysis
    email_analyzer = model.with_structured_output(EmailAnalysis)
    
    # First pass: Analyze the email with structured output
    analysis_messages = [
        {"role": "system",
         "content": """You are an advanced email analyzer. Extract key information from the email and classify its tone and urgency.
         
         Rules for classification:
         - URGENCY: 
           - HIGH: Contains words like "immediately", "urgent", "ASAP", "emergency", or deadlines within 24 hours
           - MEDIUM: Contains words suggesting timely action within days
           - LOW: General information or requests without time pressure
         
         - TONE:
           - ESCALATION: Customer is escalating an issue or requesting to speak with management
           - THREATENING: Contains legal threats, account cancellation threats, or aggressive language
           - URGENT: Communicates urgency without being threatening
           - POLITE: Courteous language regardless of the request
           - NEUTRAL: Neither particularly polite nor impolite
           
         - ACTION NEEDED:
           - ESCALATE: Any email with HIGH urgency, or with ESCALATION/THREATENING tone
           - ROUTINE: All other emails
         
         Provide a comprehensive analysis based on these criteria.
         """
         },
        {
            "role": "user",
            "content": last_message[-1].content
        }
    ]
    
    email_data = email_analyzer.invoke(analysis_messages)
    
    # Determine and execute the appropriate action
    action_result = None
    if email_data.action_needed == "escalate":
        action_result = simulate_crm_notification(email_data)
        action_message = f"⚠️ **EMAIL ESCALATED** ⚠️\nTicket created: {action_result.get('ticket_id')}"
    else:
        action_result = log_routine_email(email_data)
        action_message = f"✓ Email logged: {action_result.get('log_id')}"
    
    # Format the response for the user
    response_content = f"""
## Email Analysis

**From:** {email_data.sender}
**Urgency:** {email_data.urgency.value.upper()}
**Issue:** {email_data.issue}
**Tone Detected:** {email_data.tone.value.capitalize()}
**Action Taken:** {email_data.action_needed.capitalize()}

**Summary:** {email_data.summary}

{action_message}
    """
    
    return {"messages": [{"role": "assistant", "content": response_content}]}

def simulate_alert_logging(alert_data: dict):
    """
    Simulates logging an alert to an external system.
    In a real implementation, this would make an API call.
    """
    print(f"[ALERT LOGGED] {alert_data['alert_type']}")
    print(f"[ALERT DETAILS] {alert_data['details']}")
    return {
        "status": "logged",
        "alert_id": "ALT-" + str(hash(str(alert_data)) % 10000),
        "timestamp": "2025-05-31T12:00:00Z"
    }

# Replace the multiple webhook models with a single payment webhook model
class PaymentWebhook(BaseModel):
    transaction_id: str = Field(description="Unique identifier for the transaction")
    amount: float = Field(description="Transaction amount")
    status: str = Field(description="Transaction status (e.g., completed, pending, failed)")
    timestamp: str = Field(description="ISO format timestamp of the transaction")
    
    # Optional fields
    currency: str | None = Field(None, description="Currency code (e.g., USD, EUR)")
    payment_method: str | None = Field(None, description="Method of payment")
    customer_id: str | None = Field(None, description="Customer identifier")
    source: str | None = Field(None, description="Source of the transaction")
    campaign: str | None = Field(None, description="Marketing campaign associated with transaction")

def json_agent(state: State):
    last_message = state["messages"]
    
    # Set up structured output for JSON validation
    json_validator = model.with_structured_output(JsonValidationResult)
    
    validation_messages = [
        {"role": "system",
         "content": """You are a JSON webhook validator. Analyze the provided JSON data and validate it against our payment webhook schema.
         
         Payment webhook schema:
         - Required fields: transaction_id (string), amount (number), status (string), timestamp (ISO date string)
         - Optional fields: currency, payment_method, customer_id, source, campaign
         
         For the JSON document:
         1. Check if all required fields are present
         2. Validate data types (amount should be a number, timestamp should be a valid date format)
         3. Flag fields as anomalies ONLY if they don't belong to either required or optional fields
         4. Determine if an alert needs to be logged (missing required fields or type errors)
         """
        },
        {
            "role": "user",
            "content": last_message[-1].content
        }
    ]
    
    validation_result = json_validator.invoke(validation_messages)
    
    # Log alert if needed
    alert_logged = None
    if validation_result.alert_needed:
        alert_data = {
            "alert_type": "Payment Webhook Validation Error",
            "details": f"Missing fields: {validation_result.missing_fields}, Type errors: {validation_result.type_errors}",
            "severity": "high" if len(validation_result.missing_fields) > 0 else "medium"
        }
        alert_logged = simulate_alert_logging(alert_data)
        alert_message = f"⚠️ **ALERT LOGGED** ⚠️\nAlert ID: {alert_logged.get('alert_id')}"
    else:
        alert_message = "✓ JSON validation passed - no alerts needed"
    
    # Format the response
    response_content = f"""
## Payment Webhook Analysis

**Validation Status:** {"❌ INVALID" if not validation_result.is_valid else "✅ VALID"}

**Missing Required Fields:** {', '.join(validation_result.missing_fields) if validation_result.missing_fields else "None"}

**Type Errors:** {', '.join(validation_result.type_errors) if validation_result.type_errors else "None"}

**Other Anomalies:** {', '.join(validation_result.anomalies) if validation_result.anomalies else "None"}

**Summary:** {validation_result.summary}

{alert_message}
    """
    
    return {"messages": [{"role": "assistant", "content": response_content}]}

def pdf_agent(state: State):
    last_message = state["messages"]
    print(last_message[-1].content)
    # Set up structured output for PDF analysis
    pdf_analyzer = model.with_structured_output(PdfAnalysisResult)
    
    analysis_messages = [
        {
            "role": "system",
            "content": """You are a PDF document analyzer. Analyze the provided text to:
            1. Determine if it is an invoice, policy, or other document.
            2. Extract key fields 
            3. extract total amount as seperate field for invoices.
            4. Identify regulatory keywords (e.g., GDPR, HIPAA) for policies.
            5. Flag if the document needs attention (e.g., invoices with total > 10000 or regulatory terms in policies).
            6. Provide a brief summary.
            If a field cannot be extracted, use default values (e.g., empty dict for extracted_fields, empty list for flagged_keywords, None for total_amount).
            """
        },
        {
            "role": "user",
            "content": last_message[-1].content
        }
    ]
    
    pdf_result = pdf_analyzer.invoke(analysis_messages)
    print(pdf_result)
    # Determine if document needs attention
    attention_reason = []
    if pdf_result.document_type == "invoice" and pdf_result.total_amount and pdf_result.total_amount > 10000:
        attention_reason.append(f"High-value invoice (${pdf_result.total_amount:,.2f})")
    
    if pdf_result.document_type == "policy" and pdf_result.flagged_keywords:
        attention_reason.append(f"Contains regulatory terms: {', '.join(pdf_result.flagged_keywords)}")
    
    attention_message = ""
    if attention_reason:
        attention_message = f"⚠️ **NEEDS ATTENTION** ⚠️\nReason: {'; '.join(attention_reason)}"
    else:
        attention_message = "✓ No special attention required"
    
    # Format extracted fields for display
    extracted_fields_formatted = "\n".join([f"**{key}:** {value}" for key, value in pdf_result.extracted_fields.items()])
    
    # Format the response
    response_content = f"""
## PDF Document Analysis

**Document Type:** {pdf_result.document_type.capitalize()}

**Extracted Information:**
{extracted_fields_formatted}

{f"**Total Amount:** ${pdf_result.total_amount:,.2f}" if pdf_result.total_amount is not None else ""}

{f"**Flagged Keywords:** {', '.join(pdf_result.flagged_keywords)}" if pdf_result.flagged_keywords else ""}

**Summary:** {pdf_result.summary}

{attention_message}
    """
    
    return {"messages": [{"role": "assistant", "content": response_content}]}

def chat_agent(state: State):
    last_message = state["messages"]
    messages = [
        {"role": "system",
         "content": """you are a chat agent to interact with the user and can elaborate content"""
         },
        {
            "role": "user",
            "content": "\n".join([msg.content for msg in last_message])
        }
    ]
    response = model.invoke(messages)
    return {"messages": [{"role": "assistant", "content": response.content}]}
import requests
from typing import Dict, Any, Literal, Union
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
import os

# Initialize the LLM (same as in your main script)
model = ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.4
)

class State(TypedDict):
    messages: list

# Payload schemas for each endpoint
class EscalatePayload(BaseModel):
    sender: str = Field(description="Email sender's address")
    issue: str = Field(description="Main issue or request in the email")
    urgency: str = Field(description="Urgency level (e.g., HIGH, MEDIUM, LOW)")

class AlertPayload(BaseModel):
    alert_type: str = Field(description="Type of alert")
class CompliancePayload(BaseModel):
    document_type: str = Field(description="Type of document")
    issue: str = Field(description="issue in one word")

class ActionDecision(BaseModel):
    endpoint: Literal["/crm/escalate", "/alert/log", "/compliance/flag", "none"] = Field(
        description="The FastAPI endpoint to call or 'none' if no action is needed"
    )
    payload: Union[EscalatePayload, AlertPayload, CompliancePayload, None] = Field(
        description="The payload matching the endpoint's schema, or None if endpoint is 'none'"
    )

def action_agent(state: State) -> Dict[str, Any]:
    """
    Uses LLM to generate structured action decision with specific payload schemas and sends HTTP POST request to FastAPI endpoint.
    """
    last_message = state["messages"][-1].content
    
    # Set up structured output for action decision
    action_decider = model.with_structured_output(ActionDecision)
    
    # System prompt for LLM to decide action and payload
    system_prompt = """
    Analyze the provided message to determine the appropriate FastAPI endpoint and payload. Use the correct payload schema for the selected endpoint.

    Available endpoints and their payload schemas:
    - /crm/escalate: For emails requiring escalation (e.g., high urgency, escalation tone).
      Payload schema: {sender: str, issue: str, urgency: str}
    - /alert/log: For JSON validation errors or anomalies.
      Payload schema: {alert_type: str}
    - /compliance/flag: For PDFs with regulatory issues or high-value invoices.
      Payload schema: {document_type: str, issue: str}
    - none: If no action is required (e.g., general chat response).
      Payload: null

    Rules:
    - Select /crm/escalate if the message contains "ESCALATED" or mentions high urgency (e.g., "Urgency: HIGH").
    - Select /alert/log if the message contains "ALERT LOGGED" or mentions JSON validation issues (e.g., "Missing required fields").
    - Select /compliance/flag if the message contains "NEEDS ATTENTION" or mentions regulatory terms (e.g., "GDPR") or high-value invoices.
    - Select "none" for all other cases (e.g., general chat responses).
    - Ensure the payload matches the schema of the selected endpoint.
    """
    
    try:
        # Invoke LLM to decide action
        decision = action_decider.invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": last_message}
        ])
        print(decision)
        # Base URL for FastAPI server
        base_url = "http://localhost:8000"
        response_content = "No specific action required."
        
        # Send POST request if endpoint is not "none"
        if decision.endpoint != "none" and decision.payload is not None:
            response = requests.post(f"{base_url}{decision.endpoint}", json=decision.payload.dict())
            response.raise_for_status()
            result = response.json()
            if decision.endpoint == "/crm/escalate":
                response_content = f"Email escalated. Ticket ID: {result.get('ticket_id')}"
            elif decision.endpoint == "/alert/log":
                response_content = f"Alert logged. Alert ID: {result.get('alert_id')}"
            elif decision.endpoint == "/compliance/flag":
                response_content = f"Compliance risk flagged. Flag ID: {result.get('flag_id')}"
        elif decision.endpoint == "none":
            response_content = "No specific action required."
        else:
            response_content = "Error: No payload provided for action."
        
    except requests.RequestException as e:
        response_content = f"Failed to execute action: HTTP error - {str(e)}"
    except Exception as e:
        response_content = f"Failed to execute action: {str(e)}"
    
    return {
        "messages": [{"role": "assistant", "content": response_content}]
    }
builder = StateGraph(MessagesState)
builder.add_node(supervisor)
builder.add_node(email_agent)
builder.add_node(pdf_agent)
builder.add_node(json_agent)
builder.add_node(chat_agent)
builder.add_node(action_agent)
builder.add_edge(START, "supervisor")
builder.add_edge("pdf_agent", "action_agent")
builder.add_edge("json_agent", "action_agent")
builder.add_edge("email_agent", "action_agent")
builder.add_edge("pdf_agent", END)
builder.add_edge("email_agent", END)
builder.add_edge("json_agent", END)
builder.add_edge("chat_agent", END)
builder.add_edge("action_agent", END)
graph = builder.compile()