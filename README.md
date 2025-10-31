# 🏥 Core Care - AI-Powered Nutrition Analysis Platform

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![Flask](https://img.shields.io/badge/flask-3.0+-green.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/license-Private-lightgrey.svg)

> An intelligent nutrition analysis system powered by fine-tuned Vision-Language Models (VLMs) for food recognition and nutritional assessment.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Model Information](#model-information)
- [Development](#development)
- [Deployment](#deployment)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

Core Care is an AI-powered platform that combines computer vision and natural language processing to provide:
- **Food Recognition**: Identify foods from images using fine-tuned VLMs
- **Nutritional Analysis**: Extract detailed nutritional information
- **Interactive Q&A**: Answer nutrition-related questions with context
- **Ingredient Lookup**: Vector-based search through nutrition database

## ✨ Features

### 🤖 AI-Powered Analysis
- Fine-tuned Qwen3-VL model with LoRA adapters
- Support for both image-based and text-only queries
- Real-time inference with GPU acceleration
- Persistent model loading for fast response times

### 🔍 Intelligent Search
- Vector similarity search using Milvus
- Semantic ingredient lookup
- RAG (Retrieval-Augmented Generation) pipeline

### 🚀 Performance
- 4-bit quantization for efficient memory usage
- Cached model loading (3-5x faster inference)
- Optimized for RTX 4050 (6GB VRAM)

### 📊 Workflow Orchestration
- LangGraph-based processing pipeline
- Modular node architecture
- State management for complex workflows

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     Client Layer                        │
│            (Mobile App / API Clients)                   │
└──────────────────┬──────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────┐
│                   Flask API Layer                       │
│  • REST Endpoints                                       │
│  • Request Validation                                   │
│  • Response Formatting                                  │
└──────────────────┬──────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────┐
│                LangGraph Orchestration                  │
│  • Workflow Management                                  │
│  • State Tracking                                       │
│  • Node Coordination                                    │
└──────────────────┬──────────────────────────────────────┘
                   │
        ┌──────────┴──────────┐
        ▼                     ▼
┌──────────────────┐   ┌──────────────────┐
│  Model Service   │   │  Vector Search   │
│  • VLM Loading   │   │  • Milvus DB     │
│  • LoRA Adapter  │   │  • Embeddings    │
│  • Inference     │   │  • Similarity    │
└──────────────────┘   └──────────────────┘
```

## 🛠️ Tech Stack

### Core Framework
- **Flask 3.0+** - Web framework
- **Python 3.10+** - Programming language

### AI/ML
- **PyTorch 2.0+** - Deep learning framework
- **Transformers** - Hugging Face library
- **PEFT** - Parameter-Efficient Fine-Tuning
- **BitsAndBytes** - Model quantization

### Models
- **Base**: Qwen2.5-VL-4B-Instruct (4-bit quantized)
- **Fine-tuned**: Custom LoRA adapters for nutrition domain

### Vector Database
- **Milvus** - Vector similarity search
- **Sentence Transformers** - Embedding generation

### Workflow
- **LangGraph** - Workflow orchestration
- **LangChain** - LLM application framework

### Development
- **Git** - Version control
- **Docker** - Containerization (optional)
- **pytest** - Testing framework

## 📦 Installation

### Prerequisites

```bash
# System requirements
- Python 3.10 or higher
- CUDA 12.1+ (for GPU support)
- 16GB+ RAM
- 6GB+ VRAM (for RTX 4050 or similar)

# Optional
- Milvus (for vector search)
- Docker (for containerized deployment)
```

### Step 1: Clone Repository

```bash
git clone https://github.com/yourusername/Core-Care.git
cd Core-Care
```

### Step 2: Create Virtual Environment

```bash
# Windows
python -m venv CoreCareVenv
CoreCareVenv\Scripts\activate

# Linux/Mac
python3 -m venv CoreCareVenv
source CoreCareVenv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Configure Environment

```bash
# Copy example env file
cp .env.example .env

# Edit .env with your settings
# See Configuration section below
```

### Step 5: Run Application

```bash
python run.py
```

The application will be available at `http://localhost:5000`

## ⚙️ Configuration

Create a `.env` file in the root directory:

```bash
# Flask Configuration
FLASK_ENV=development
FLASK_DEBUG=True
SECRET_KEY=your-secret-key-here

# Server
HOST=0.0.0.0
PORT=5000

# Model Configuration
MODEL_ID=unsloth/Qwen3-VL-4B-Instruct-bnb-4bit
LORA_ADAPTER_ID=WissMah/Qwen2.5VL-FT-Lora_mix_aug1

# Model Settings
MAX_NEW_TOKENS=256
MAX_IMAGE_SIZE=448

# Milvus Configuration
MILVUS_HOST=localhost
MILVUS_PORT=19530
MILVUS_COLLECTION=nutrition_ingredients

# Logging
LOG_LEVEL=INFO
LOG_FILE=app.log

# Upload Settings
MAX_UPLOAD_SIZE=10485760  # 10MB
ALLOWED_EXTENSIONS=jpg,jpeg,png,gif,webp
```

### Important Notes

⚠️ **Never commit `.env` files to Git!**

The `.env` file contains sensitive information and is already in `.gitignore`.

## 🚀 Usage

### Basic Example - Text Query

```python
import requests

response = requests.post(
    'http://localhost:5000/api/analyze',
    json={
        'prompt': 'What is protein?',
        'max_new_tokens': 100
    }
)

print(response.json())
```

### Image Analysis

```python
import requests
import base64

# Read and encode image
with open('food_image.jpg', 'rb') as f:
    image_data = base64.b64encode(f.read()).decode('utf-8')

response = requests.post(
    'http://localhost:5000/api/analyze',
    json={
        'prompt': 'What food is this and what are its nutritional values?',
        'image': image_data,
        'max_new_tokens': 200
    }
)

print(response.json())
```

## 📚 API Documentation

### Endpoints

#### POST /api/analyze

Analyze food images or answer nutrition questions.

**Request:**
```json
{
  "prompt": "What nutrients are in this food?",
  "image": "base64_encoded_image_data",  // Optional
  "max_new_tokens": 256  // Optional, default: 256
}
```

**Response:**
```json
{
  "response": "This food contains...",
  "processing_time": 1.234,
  "model_used": "base+lora"
}
```

#### GET /api/health

Check API health status.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "timestamp": "2025-10-27T10:30:00Z"
}
```

## 🤖 Model Information

### Base Model

**Name**: Qwen2.5-VL-4B-Instruct  
**Quantization**: 4-bit (BitsAndBytes)  
**VRAM Usage**: ~2.5 GB  
**Provider**: Unsloth (optimized version)

### LoRA Adapter

**Name**: Qwen2.5VL-FT-Lora_mix_aug1  
**Type**: Parameter-Efficient Fine-Tuning  
**Domain**: Nutrition & Food Recognition  
**Additional VRAM**: ~100 MB

### Model Loading Strategy

- **Startup**: Load base model + LoRA adapter once
- **Text Queries**: Use loaded model (instant)
- **Image Queries**: Use same loaded model (instant)
- **Speed**: 3-5x faster than per-request loading

## 👨‍💻 Development

### Project Structure

```
Core-Care/
├── app/
│   ├── __init__.py
│   ├── routes/
│   │   └── api_routes.py
│   ├── services/
│   │   ├── model_service.py      # Model loading & inference
│   │   ├── langgraph_service.py  # Workflow orchestration
│   │   └── milvus_service.py     # Vector search
│   ├── graphs/
│   │   └── nodes.py               # LangGraph nodes
│   └── utils/
│       └── helpers.py
├── tests/
│   ├── test_model_service.py
│   ├── test_api.py
│   └── test_workflow.py
├── config/
│   └── settings.py
├── logs/
│   └── app.log
├── .env                           # Secret config (not in Git)
├── .env.example                   # Example config
├── .gitignore
├── requirements.txt
├── README.md
└── run.py                         # Application entry point
```

### Running Tests

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run all tests
pytest

# Run with coverage
pytest --cov=app tests/

# Run specific test file
pytest tests/test_model_service.py
```

### Code Style

This project follows:
- **PEP 8** - Python style guide
- **Type hints** - For better code documentation
- **Docstrings** - Google style

```bash
# Format code
pip install black
black app/

# Check style
pip install flake8
flake8 app/

# Type checking
pip install mypy
mypy app/
```

### Git Workflow

We use **Git Flow** branching strategy:

```
main (production-ready)
  └── development v1 (active development)
      ├── feature/food-recognition
      ├── feature/nutrition-db
      └── bugfix/empty-responses
```

### Branch Naming Convention

- `feature/feature-name` - New features
- `bugfix/bug-name` - Bug fixes
- `hotfix/critical-fix` - Urgent fixes
- `refactor/component-name` - Code improvements
- `docs/documentation-update` - Documentation changes

## 🚢 Deployment

### Docker Deployment (Recommended)

```bash
# Build image
docker build -t core-care:latest .

# Run container
docker run -d \
  --name core-care \
  --gpus all \
  -p 5000:5000 \
  -v $(pwd)/.env:/app/.env \
  core-care:latest
```

### Production Considerations

- Use **Gunicorn** or **uWSGI** instead of Flask development server
- Set up **NGINX** as reverse proxy
- Enable **HTTPS** with SSL certificates
- Configure **environment variables** securely
- Set up **monitoring** (Prometheus, Grafana)
- Implement **logging** (ELK stack)
- Use **Docker Compose** for multi-container setup

## 🤝 Contributing

This is a private repository. For internal contributors:

1. Create a feature branch from `development v1`
2. Make your changes
3. Write/update tests
4. Submit a pull request
5. Wait for code review

### Commit Message Convention

```
type(scope): subject

body (optional)

footer (optional)
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Formatting
- `refactor`: Code restructuring
- `test`: Adding tests
- `chore`: Maintenance

**Example:**
```
feat(model): add LoRA adapter support

- Implemented unified LoRA loading
- Model stays in memory for faster inference
- Added comprehensive logging

Closes #123
```

## 📄 License

This is a private project. All rights reserved.

## 👥 Authors

- **Development Team** - Core Care Development

## 🙏 Acknowledgments

- Hugging Face for model hosting
- Anthropic for guidance
- Open source community

## 📞 Support

For issues and questions:
- **Internal**: Contact development team
- **Issues**: Use GitHub Issues (private repo)

---

**Built with ❤️ by the Core Care Team**
