from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime
import hashlib
import sqlite3
from contextlib import contextmanager
import uuid

app = FastAPI()

# SQLite database setup
DATABASE = "actions.db"

def init_db():
    """Initialize SQLite database and create tables."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        # Create crm_escalations table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS crm_escalations (
                id TEXT PRIMARY KEY,
                sender TEXT,
                issue TEXT,
                urgency TEXT,
                timestamp TEXT
            )
        """)
        # Create alert_logs table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS alert_logs (
                id TEXT PRIMARY KEY,
                alert_type TEXT,
                timestamp TEXT
            )
        """)
        # Create compliance_flags table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS compliance_flags (
                id TEXT PRIMARY KEY,
                document_type TEXT,
                issue TEXT,
                timestamp TEXT
            )
        """)
        conn.commit()

@contextmanager
def get_db_connection():
    """Context manager for SQLite database connections."""
    conn = sqlite3.connect(DATABASE)
    try:
        yield conn
    finally:
        conn.close()

# Initialize database on startup
@app.on_event("startup")
async def startup_event():
    init_db()

class EscalateRequest(BaseModel):
    sender: str
    issue: str
    urgency: str

class AlertRequest(BaseModel):
    alert_type: str

class ComplianceRequest(BaseModel):
    document_type: str
    issue: str

@app.post("/crm/escalate")
async def escalate_crm(request: EscalateRequest):
    ticket_id = f"CRM-{uuid.uuid4().hex[:8]}"
    timestamp = datetime.utcnow().isoformat() + "Z"
    
    # Store in SQLite
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO crm_escalations (id, sender, issue, urgency, timestamp)
            VALUES (?, ?, ?, ?, ?)
        """, (ticket_id, request.sender, request.issue, request.urgency, timestamp))
        conn.commit()
    
    return {
        "status": "success",
        "ticket_id": ticket_id,
        "timestamp": timestamp
    }

@app.post("/alert/log")
async def log_alert(request: AlertRequest):
    alert_id = f"ALT-{uuid.uuid4().hex[:8]}"
    timestamp = datetime.utcnow().isoformat() + "Z"
    
    # Store in SQLite
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO alert_logs (id, alert_type, timestamp)
            VALUES (?, ?, ?)
        """, (alert_id, request.alert_type, timestamp))
        conn.commit()
    
    return {
        "status": "logged",
        "alert_id": alert_id,
        "timestamp": timestamp
    }

@app.post("/compliance/flag")
async def flag_compliance(request: ComplianceRequest):
    flag_id = f"FLG-{uuid.uuid4().hex[:8]}"
    timestamp = datetime.utcnow().isoformat() + "Z"
    
    # Store in SQLite
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO compliance_flags (id, document_type, issue, timestamp)
            VALUES (?, ?, ?, ?)
        """, (flag_id, request.document_type, request.issue, timestamp))
        conn.commit()
    
    return {
        "status": "flagged",
        "flag_id": flag_id,
        "timestamp": timestamp
    }