"""
Disaster QA API - FastAPI Implementation

RESTful API endpoints for disaster question answering system
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import logging
import asyncio
from datetime import datetime
import uvicorn

from ..modules.qa_pipeline import DisasterQAPipeline, QAResult, QuestionType
from ..utils import load_config, setup_logging

# ログ設定
logger = logging.getLogger(__name__)

# Pydantic Models
class QARequest(BaseModel):
    """QA リクエストモデル"""
    question: str = Field(..., min_length=1, max_length=500, 
                         description="災害に関する質問")
    language: str = Field(default="auto", 
                         description="言語設定 ('ja', 'en', 'auto')")
    top_k: Optional[int] = Field(default=None, ge=1, le=10,
                               description="検索する文書数")

class QAResponse(BaseModel):
    """QA レスポンスモデル"""
    answer: str
    confidence: float = Field(ge=0.0, le=1.0)
    question_type: str
    disaster_category: str
    normalized_query: str
    sources: List[Dict[str, Any]]
    processing_time_ms: float
    timestamp: datetime

class BatchQARequest(BaseModel):
    """バッチQA リクエストモデル"""
    questions: List[str] = Field(..., min_items=1, max_items=20)
    language: str = Field(default="auto")
    top_k: Optional[int] = Field(default=None, ge=1, le=10)

class BatchQAResponse(BaseModel):
    """バッチQA レスポンスモデル"""
    results: List[QAResponse]
    total_questions: int
    total_processing_time_ms: float
    timestamp: datetime

class HealthResponse(BaseModel):
    """ヘルスチェック レスポンス"""
    status: str
    version: str
    pipeline_info: Dict[str, Any]
    timestamp: datetime

class ErrorResponse(BaseModel):
    """エラー レスポンス"""
    error: str
    detail: str
    timestamp: datetime

# FastAPI アプリ作成
def create_app() -> FastAPI:
    """FastAPI アプリケーションを作成"""
    
    config = load_config()
    setup_logging(config)
    
    app = FastAPI(
        title="Disaster Question Answering API",
        description="災害質問応答システム REST API - MVP版",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    # CORS設定
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # 本番環境では制限
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # グローバル変数としてパイプライン保持
    app.state.qa_pipeline = None
    app.state.config = config
    
    return app

app = create_app()

# Startup Event
@app.on_event("startup")
async def startup_event():
    """アプリ起動時の初期化処理"""
    logger.info("Starting Disaster QA API...")
    
    try:
        # QAパイプライン初期化 (バックグラウンドで実行)
        loop = asyncio.get_event_loop()
        app.state.qa_pipeline = await loop.run_in_executor(
            None, DisasterQAPipeline, app.state.config
        )
        logger.info("QA Pipeline initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize QA Pipeline: {e}")
        app.state.qa_pipeline = None

# Shutdown Event  
@app.on_event("shutdown")
async def shutdown_event():
    """アプリ終了時の処理"""
    logger.info("Shutting down Disaster QA API...")

# Helper Functions
def qa_result_to_response(qa_result: QAResult, processing_time: float) -> QAResponse:
    """QAResult を QAResponse に変換"""
    return QAResponse(
        answer=qa_result.answer,
        confidence=qa_result.confidence,
        question_type=qa_result.question_type.value,
        disaster_category=qa_result.disaster_category,
        normalized_query=qa_result.normalized_query,
        sources=qa_result.sources,
        processing_time_ms=processing_time,
        timestamp=datetime.now()
    )

# API Endpoints

@app.get("/", response_model=Dict[str, str])
async def root():
    """ルートエンドポイント"""
    return {
        "message": "Disaster Question Answering API",
        "version": "0.1.0",
        "docs": "/docs"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """ヘルスチェックエンドポイント"""
    
    pipeline_info = {}
    status = "unhealthy"
    
    if app.state.qa_pipeline is not None:
        try:
            pipeline_info = app.state.qa_pipeline.get_pipeline_info()
            status = "healthy"
        except Exception as e:
            pipeline_info = {"error": str(e)}
    
    return HealthResponse(
        status=status,
        version="0.1.0",
        pipeline_info=pipeline_info,
        timestamp=datetime.now()
    )

@app.post("/qa", response_model=QAResponse)
async def answer_question(request: QARequest):
    """単一質問に対する回答エンドポイント"""
    
    if app.state.qa_pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="QA Pipeline not available. Check /health endpoint."
        )
    
    start_time = datetime.now()
    
    try:
        # バックグラウンドでQA実行
        loop = asyncio.get_event_loop()
        qa_result = await loop.run_in_executor(
            None,
            app.state.qa_pipeline.answer_question,
            request.question,
            request.language,
            request.top_k
        )
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return qa_result_to_response(qa_result, processing_time)
        
    except Exception as e:
        logger.error(f"QA processing error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@app.post("/qa/batch", response_model=BatchQAResponse)
async def answer_questions_batch(request: BatchQARequest):
    """バッチ質問に対する回答エンドポイント"""
    
    if app.state.qa_pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="QA Pipeline not available. Check /health endpoint."
        )
    
    start_time = datetime.now()
    
    try:
        # バックグラウンドでバッチQA実行
        loop = asyncio.get_event_loop()
        qa_results = await loop.run_in_executor(
            None,
            app.state.qa_pipeline.batch_answer,
            request.questions
        )
        
        total_processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # 個別結果変換
        responses = []
        for qa_result in qa_results:
            response = qa_result_to_response(qa_result, 0)  # 個別時間は0
            responses.append(response)
        
        return BatchQAResponse(
            results=responses,
            total_questions=len(request.questions),
            total_processing_time_ms=total_processing_time,
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Batch QA processing error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@app.get("/categories", response_model=List[str])
async def get_disaster_categories():
    """利用可能な災害カテゴリ一覧"""
    return [
        "earthquake", "tsunami", "typhoon", "flood", 
        "volcanic_eruption", "landslide", "general"
    ]

@app.get("/question-types", response_model=List[str])
async def get_question_types():
    """利用可能な質問タイプ一覧"""
    return [qtype.value for qtype in QuestionType]

@app.post("/glossary/normalize", response_model=Dict[str, Any])
async def normalize_query(query: str):
    """用語正規化エンドポイント"""
    
    if app.state.qa_pipeline is None or not hasattr(app.state.qa_pipeline, 'glossary'):
        raise HTTPException(
            status_code=503,
            detail="Glossary service not available"
        )
    
    try:
        normalization_result = app.state.qa_pipeline.glossary.normalize_query(query)
        return normalization_result
        
    except Exception as e:
        logger.error(f"Normalization error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Normalization failed: {str(e)}"
        )

# Exception Handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    """HTTP例外ハンドラー"""
    return ErrorResponse(
        error=f"HTTP {exc.status_code}",
        detail=exc.detail,
        timestamp=datetime.now()
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc: Exception):
    """一般例外ハンドラー"""
    logger.error(f"Unhandled exception: {exc}")
    return ErrorResponse(
        error="Internal Server Error",
        detail=str(exc),
        timestamp=datetime.now()
    )

# メイン実行関数
def main():
    """APIサーバー起動"""
    
    config = load_config()
    api_config = config.get('api', {})
    
    host = api_config.get('host', '127.0.0.1')
    port = api_config.get('port', 8000)
    debug = api_config.get('debug', False)
    
    logger.info(f"Starting Disaster QA API server on {host}:{port}")
    
    uvicorn.run(
        "src.api.main:app",
        host=host,
        port=port,
        reload=debug,
        log_level="info" if not debug else "debug"
    )

if __name__ == "__main__":
    main()