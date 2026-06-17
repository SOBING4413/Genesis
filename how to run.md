# 1. Clone & setup
git clone https://github.com/your-org/genesis-ai.git
cd genesis-ai
python -m venv venv
source venv/bin/activate  # Linux/Mac
pip install -r requirements.txt

# 2. Start infrastructure
docker-compose up -d postgres neo4j redis

# 3. Run API
uvicorn app.main:app --reload

# 4. Di terminal lain, jalankan Celery worker
celery -A app.tasks.celery_app worker --loglevel=info

# 5. Buka http://localhost:8000
#    Docs: http://localhost:8000/docs