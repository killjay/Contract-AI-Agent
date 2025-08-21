"""
FastAPI application for Legal Document Review AI Agent.
"""

import os
import asyncio
import uuid
from typing import Optional, List
from pathlib import Path
import shutil

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel

from src.core.models import (
    DocumentType, ReviewResult, RiskAssessment, ReviewRequest
)
from src.core.config import get_config
from src.agent import LegalDocumentReviewAgent


# Initialize FastAPI app
app = FastAPI(
    title="Legal Document Review AI Agent",
    description="AI-powered legal document analysis and review system",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global configuration and agent
config = get_config()
agent = LegalDocumentReviewAgent()

# In-memory storage for demo (use database in production)
active_workflows = {}
completed_reviews = {}


class UploadResponse(BaseModel):
    """Response model for file upload."""
    file_id: str
    filename: str
    file_size: int
    message: str


class AnalysisRequest(BaseModel):
    """Request model for document analysis."""
    file_id: str
    document_type: Optional[DocumentType] = None
    priority_areas: Optional[List[str]] = []
    custom_instructions: Optional[str] = None


class WorkflowStatusResponse(BaseModel):
    """Response model for workflow status."""
    workflow_id: str
    status: str
    progress: int
    current_step: str
    estimated_completion: Optional[str] = None


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Legal Document Review AI Agent API",
        "version": "1.0.0",
        "status": "active",
        "endpoints": {
            "upload": "/upload",
            "analyze": "/analyze",
            "status": "/status/{workflow_id}",
            "result": "/result/{workflow_id}",
            "health": "/health"
        }
    }


@app.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a legal document for analysis.
    
    Supported formats: PDF, DOCX, DOC, TXT
    """
    # Validate file type
    allowed_extensions = config.storage.allowed_file_types
    file_extension = Path(file.filename).suffix.lower().lstrip('.')
    
    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}"
        )
    
    # Validate file size
    file_size = 0
    content = await file.read()
    file_size = len(content)
    
    if file_size > config.storage.max_file_size_mb * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size: {config.storage.max_file_size_mb}MB"
        )
    
    # Create upload directory if it doesn't exist
    upload_dir = Path(config.storage.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate unique file ID and save file
    file_id = str(uuid.uuid4())
    file_path = upload_dir / f"{file_id}_{file.filename}"
    
    with open(file_path, "wb") as buffer:
        buffer.write(content)
    
    return UploadResponse(
        file_id=file_id,
        filename=file.filename,
        file_size=file_size,
        message="File uploaded successfully"
    )


@app.post("/analyze")
async def analyze_document(
    request: AnalysisRequest,
    background_tasks: BackgroundTasks
):
    """
    Start analysis of an uploaded document.
    
    Returns a workflow ID to track progress.
    """
    # Find uploaded file
    upload_dir = Path(config.storage.upload_dir)
    file_pattern = f"{request.file_id}_*"
    matching_files = list(upload_dir.glob(file_pattern))
    
    if not matching_files:
        raise HTTPException(
            status_code=404,
            detail="File not found. Please upload the file first."
        )
    
    file_path = matching_files[0]
    workflow_id = str(uuid.uuid4())
    
    # Store workflow info
    active_workflows[workflow_id] = {
        "status": "queued",
        "progress": 0,
        "current_step": "Initializing analysis",
        "file_path": str(file_path),
        "request": request
    }
    
    # Start analysis in background
    background_tasks.add_task(
        run_analysis,
        workflow_id,
        str(file_path),
        request
    )
    
    return {
        "workflow_id": workflow_id,
        "status": "started",
        "message": "Document analysis started. Use the workflow ID to check progress."
    }


@app.get("/status/{workflow_id}", response_model=WorkflowStatusResponse)
async def get_workflow_status(workflow_id: str):
    """Get the status of a running analysis workflow."""
    
    # Check if workflow is completed
    if workflow_id in completed_reviews:
        return WorkflowStatusResponse(
            workflow_id=workflow_id,
            status="completed",
            progress=100,
            current_step="Analysis completed"
        )
    
    # Check if workflow is active
    if workflow_id in active_workflows:
        workflow = active_workflows[workflow_id]
        return WorkflowStatusResponse(
            workflow_id=workflow_id,
            status=workflow["status"],
            progress=workflow["progress"],
            current_step=workflow["current_step"]
        )
    
    raise HTTPException(
        status_code=404,
        detail="Workflow not found"
    )


@app.get("/result/{workflow_id}")
async def get_analysis_result(workflow_id: str):
    """Get the complete analysis result for a workflow."""
    
    if workflow_id not in completed_reviews:
        raise HTTPException(
            status_code=404,
            detail="Analysis not completed or workflow not found"
        )
    
    return completed_reviews[workflow_id]


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Check LLM connectivity
        llm_health = await agent.llm_client.health_check()
        
        # Check configuration
        config_validation = config.validate_config()
        
        return {
            "status": "healthy" if config_validation["valid"] else "degraded",
            "llm_providers": llm_health,
            "config_validation": config_validation,
            "active_workflows": len(active_workflows),
            "completed_reviews": len(completed_reviews)
        }
    
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "error": str(e)}
        )


@app.post("/quick-risk-assessment")
async def quick_risk_assessment(request: AnalysisRequest):
    """Perform a quick risk assessment without full analysis."""
    
    # Find uploaded file
    upload_dir = Path(config.storage.upload_dir)
    file_pattern = f"{request.file_id}_*"
    matching_files = list(upload_dir.glob(file_pattern))
    
    if not matching_files:
        raise HTTPException(
            status_code=404,
            detail="File not found. Please upload the file first."
        )
    
    file_path = matching_files[0]
    
    try:
        risk_assessment = await agent.quick_risk_assessment(str(file_path))
        return risk_assessment
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Risk assessment failed: {str(e)}"
        )


@app.get("/download/{workflow_id}")
async def download_redlined_document(workflow_id: str):
    """Download the redlined document (if available)."""
    
    if workflow_id not in completed_reviews:
        raise HTTPException(
            status_code=404,
            detail="Analysis not completed or workflow not found"
        )
    
    # In a full implementation, this would generate and return a Word document
    # with tracked changes. For now, return a placeholder.
    raise HTTPException(
        status_code=501,
        detail="Document download not implemented yet"
    )


@app.delete("/cleanup/{workflow_id}")
async def cleanup_workflow(workflow_id: str):
    """Clean up files and data for a completed workflow."""
    
    # Remove from active workflows
    if workflow_id in active_workflows:
        del active_workflows[workflow_id]
    
    # Remove from completed reviews
    if workflow_id in completed_reviews:
        del completed_reviews[workflow_id]
    
    # Clean up uploaded files (optional)
    # Implementation depends on file retention policy
    
    return {"message": "Workflow cleaned up successfully"}


async def run_analysis(
    workflow_id: str,
    file_path: str,
    request: AnalysisRequest
):
    """Run the document analysis in the background."""
    
    try:
        # Update workflow status
        active_workflows[workflow_id]["status"] = "running"
        active_workflows[workflow_id]["progress"] = 10
        active_workflows[workflow_id]["current_step"] = "Starting analysis"
        
        # Perform the analysis
        result = await agent.review_document(
            document_path=file_path,
            document_type=request.document_type,
            custom_instructions=request.custom_instructions,
            priority_areas=request.priority_areas
        )
        
        # Update progress during analysis (would need agent integration for real-time updates)
        active_workflows[workflow_id]["progress"] = 50
        active_workflows[workflow_id]["current_step"] = "Analyzing clauses"
        
        await asyncio.sleep(1)  # Simulate processing time
        
        active_workflows[workflow_id]["progress"] = 80
        active_workflows[workflow_id]["current_step"] = "Generating redlines"
        
        await asyncio.sleep(1)  # Simulate processing time
        
        # Complete the analysis
        active_workflows[workflow_id]["progress"] = 100
        active_workflows[workflow_id]["current_step"] = "Analysis completed"
        active_workflows[workflow_id]["status"] = "completed"
        
        # Store the result
        completed_reviews[workflow_id] = result.dict()
        
        # Remove from active workflows
        del active_workflows[workflow_id]
        
    except Exception as e:
        # Handle errors
        active_workflows[workflow_id]["status"] = "failed"
        active_workflows[workflow_id]["current_step"] = f"Error: {str(e)}"
        print(f"Analysis failed for workflow {workflow_id}: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "src.api.main:app",
        host=config.api.host,
        port=config.api.port,
        reload=config.debug,
        workers=config.api.workers
    )
