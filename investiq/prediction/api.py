"""
FastAPI application for InvestIQ prediction endpoints.
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from typing import List, Optional, Dict, Any
import logging
import uvicorn
from pathlib import Path

from .predictor import predict_return, batch_predict, get_prediction_summary

logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="InvestIQ API",
    description="AI-powered investment prediction API",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "InvestIQ API",
        "version": "0.1.0",
        "endpoints": {
            "predict": "/score/{ticker}",
            "batch_predict": "/batch",
            "health": "/health",
            "docs": "/docs"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "InvestIQ API"}


@app.get("/score/{ticker}")
async def get_prediction(
    ticker: str,
    horizon: int = Query(5, ge=1, le=30, description="Prediction horizon in days"),
    confidence_threshold: float = Query(0.6, ge=0.0, le=1.0, description="Confidence threshold"),
    model_path: Optional[str] = Query(None, description="Path to specific model")
):
    """
    Get prediction for a single ticker.
    
    Args:
        ticker: Stock ticker symbol
        horizon: Prediction horizon in days (1-30)
        confidence_threshold: Confidence threshold (0.0-1.0)
        model_path: Optional path to specific model
        
    Returns:
        Prediction result with suggestion
    """
    try:
        result = predict_return(
            ticker=ticker,
            model_path=model_path,
            horizon=horizon,
            confidence_threshold=confidence_threshold
        )
        
        if 'error' in result:
            raise HTTPException(status_code=400, detail=result['error'])
        
        return result
        
    except Exception as e:
        logger.error(f"Error in prediction endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch")
async def batch_predictions(
    tickers: List[str],
    horizon: int = Query(5, ge=1, le=30, description="Prediction horizon in days"),
    model_path: Optional[str] = Query(None, description="Path to specific model")
):
    """
    Get predictions for multiple tickers.
    
    Args:
        tickers: List of stock ticker symbols
        horizon: Prediction horizon in days (1-30)
        model_path: Optional path to specific model
        
    Returns:
        Dictionary of prediction results
    """
    try:
        if not tickers:
            raise HTTPException(status_code=400, detail="No tickers provided")
        
        if len(tickers) > 50:  # Limit batch size
            raise HTTPException(status_code=400, detail="Too many tickers (max 50)")
        
        results = batch_predict(
            tickers=tickers,
            model_path=model_path,
            horizon=horizon
        )
        
        # Add summary
        summary = get_prediction_summary(results)
        
        return {
            "results": results,
            "summary": summary,
            "tickers": tickers,
            "horizon": horizon
        }
        
    except Exception as e:
        logger.error(f"Error in batch prediction endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ui", response_class=HTMLResponse)
async def prediction_ui():
    """Simple HTML interface for predictions."""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>InvestIQ Prediction Interface</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
            .container { background: #f5f5f5; padding: 20px; border-radius: 8px; }
            input, select, button { padding: 10px; margin: 5px; border: 1px solid #ddd; border-radius: 4px; }
            button { background: #007bff; color: white; cursor: pointer; }
            button:hover { background: #0056b3; }
            .result { margin-top: 20px; padding: 15px; background: white; border-radius: 4px; }
            .error { background: #f8d7da; color: #721c24; }
            .success { background: #d4edda; color: #155724; }
        </style>
    </head>
    <body>
        <h1>InvestIQ Prediction Interface</h1>
        <div class="container">
            <h2>Single Ticker Prediction</h2>
            <input type="text" id="ticker" placeholder="Enter ticker symbol (e.g., AAPL)" />
            <select id="horizon">
                <option value="1">1 Day</option>
                <option value="5" selected>5 Days</option>
                <option value="10">10 Days</option>
                <option value="30">30 Days</option>
            </select>
            <button onclick="predictSingle()">Predict</button>
            
            <h2>Batch Prediction</h2>
            <input type="text" id="tickers" placeholder="Enter tickers separated by commas (e.g., AAPL,MSFT,GOOGL)" style="width: 300px;" />
            <button onclick="predictBatch()">Predict All</button>
            
            <div id="result" class="result" style="display: none;"></div>
        </div>

        <script>
            async function predictSingle() {
                const ticker = document.getElementById('ticker').value.trim().toUpperCase();
                const horizon = document.getElementById('horizon').value;
                
                if (!ticker) {
                    showResult('Please enter a ticker symbol', 'error');
                    return;
                }
                
                try {
                    const response = await fetch(`/score/${ticker}?horizon=${horizon}`);
                    const data = await response.json();
                    
                    if (response.ok) {
                        const result = `
                            <h3>Prediction for ${data.ticker}</h3>
                            <p><strong>Predicted ${horizon}D Return:</strong> ${(data.prediction * 100).toFixed(2)}%</p>
                            <p><strong>Suggestion:</strong> ${data.suggestion}</p>
                            <p><strong>Confidence:</strong> ${(data.confidence * 100).toFixed(1)}%</p>
                            <p><strong>Model:</strong> ${data.model_type}</p>
                        `;
                        showResult(result, 'success');
                    } else {
                        showResult(`Error: ${data.detail}`, 'error');
                    }
                } catch (error) {
                    showResult(`Error: ${error.message}`, 'error');
                }
            }
            
            async function predictBatch() {
                const tickersInput = document.getElementById('tickers').value.trim();
                const horizon = document.getElementById('horizon').value;
                
                if (!tickersInput) {
                    showResult('Please enter ticker symbols', 'error');
                    return;
                }
                
                const tickers = tickersInput.split(',').map(t => t.trim().toUpperCase()).filter(t => t);
                
                if (tickers.length === 0) {
                    showResult('Please enter valid ticker symbols', 'error');
                    return;
                }
                
                try {
                    const response = await fetch('/batch', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(tickers)
                    });
                    const data = await response.json();
                    
                    if (response.ok) {
                        let result = '<h3>Batch Predictions</h3>';
                        result += `<p><strong>Total:</strong> ${data.summary.total_tickers} tickers</p>`;
                        result += `<p><strong>Valid:</strong> ${data.summary.valid_predictions} predictions</p>`;
                        result += `<p><strong>Average Return:</strong> ${(data.summary.avg_prediction * 100).toFixed(2)}%</p>`;
                        result += '<h4>Individual Results:</h4><ul>';
                        
                        for (const [ticker, tickerResult] of Object.entries(data.results)) {
                            if (tickerResult.error) {
                                result += `<li>${ticker}: ERROR - ${tickerResult.error}</li>`;
                            } else {
                                const predPct = (tickerResult.prediction * 100).toFixed(2);
                                result += `<li>${ticker}: ${predPct}% - ${tickerResult.suggestion}</li>`;
                            }
                        }
                        result += '</ul>';
                        
                        showResult(result, 'success');
                    } else {
                        showResult(`Error: ${data.detail}`, 'error');
                    }
                } catch (error) {
                    showResult(`Error: ${error.message}`, 'error');
                }
            }
            
            function showResult(content, type) {
                const resultDiv = document.getElementById('result');
                resultDiv.innerHTML = content;
                resultDiv.className = `result ${type}`;
                resultDiv.style.display = 'block';
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    return app


def run_server(
    host: str = "127.0.0.1",
    port: int = 8000,
    debug: bool = True
):
    """Run the API server."""
    logger.info(f"Starting InvestIQ API server on {host}:{port}")
    uvicorn.run(
        "investiq.prediction.api:app",
        host=host,
        port=port,
        reload=debug,
        log_level="info" if not debug else "debug"
    )


if __name__ == "__main__":
    run_server()
