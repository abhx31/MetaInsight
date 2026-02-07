# MetaInsight Backend

Multi-agent document intelligence API built with FastAPI.

## 🚀 Features

- **4 AI Agents**: Orchestrator, Summarizer, Task Master, Risk Detector
- **Document Analysis**: Chunking, embedding, semantic retrieval
- **RESTful API**: Full CRUD endpoints with OpenAPI documentation
- **Async Processing**: Background job support for long documents
- **File Upload**: Support for TXT and MD files

## 📦 Installation

```bash
# Navigate to backend directory
cd apps/backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## ⚙️ Configuration

1. Copy the environment template:
```bash
cp .env.example .env
```

2. Edit `.env` and add your OpenAI API key:
```env
OPENAI_API_KEY=your-api-key-here
```

## 🏃 Running the Server

```bash
# Development mode with auto-reload
uvicorn app.main:app --reload --port 8000

# Production mode
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## 📖 API Documentation

Once running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 🔌 API Endpoints

### Health
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API info |
| GET | `/health` | Health check |
| GET | `/agents` | List all agents |

### Analysis
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/analyze` | Full document analysis |
| POST | `/analyze/async` | Async analysis (returns job_id) |
| GET | `/jobs/{job_id}` | Get async job status |

### Individual Agents
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/analyze/risks` | Risk detection only |
| POST | `/analyze/tasks` | Task extraction only |
| POST | `/analyze/summary` | Summary generation only |

### File Upload
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/upload` | Upload document file |
| POST | `/upload/analyze` | Upload and analyze |

## 📁 Project Structure

```
apps/backend/
├── requirements.txt       # Python dependencies
├── .env.example          # Environment template
├── app/
│   ├── __init__.py       # Package exports
│   ├── main.py           # FastAPI application
│   ├── config.py         # Settings management
│   ├── orchestrator.py   # Pipeline coordinator
│   ├── chunking.py       # Document segmentation
│   ├── memory.py         # Vector store (ChromaDB)
│   ├── routers.py        # API route organization
│   └── agents/
│       ├── __init__.py
│       ├── summary_agent.py    # Summarizer (placeholder)
│       ├── action_agent.py     # Task Master (placeholder)
│       ├── risk_agent.py       # Risk Agent wrapper
│       └── risk_detector/      # Full Risk Detection Agent
│           ├── agent.py        # Main agent class
│           ├── schemas.py      # Pydantic models
│           ├── scoring.py      # Risk scoring algorithm
│           ├── prompts.py      # LLM prompts
│           └── detectors/
│               ├── linguistic.py
│               ├── contextual.py
│               └── enrichment.py
```

## 🧪 Testing

```bash
# Run tests
pytest

# Run with coverage
pytest --cov=app
```

## 📝 Example Usage

```python
import requests

# Analyze a document
response = requests.post(
    "http://localhost:8000/analyze",
    json={
        "document": "Project meeting notes: We need to deliver the MVP by Q1...",
        "chunk_size": 500,
        "overlap": 100,
        "top_k": 5
    }
)

result = response.json()
print(result["risks"])
print(result["actions"])
print(result["summary"])
```

## 🔧 Development

```bash
# Format code
black app/

# Lint
flake8 app/

# Type checking
mypy app/
```

## 📄 License

MIT
