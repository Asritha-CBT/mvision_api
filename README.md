# python -m venv venv
# venv\Scripts\activate
# pip install -r requirements.txt
# pip install --no-build-isolation "git+https://github.com/KaiyangZhou/deep-person-reid.git  (cmd)
# pip install --no-build-isolation git+https://github.com/KaiyangZhou/deep-person-reid.git   (powershell)
# uvicorn mvision.main:app --reload    or  uvicorn mvision.main:app --host 0.0.0.0 --port 8000 --reload

# alembic migration
    1. alembic revision --autogenerate -m "migration message"
    2. alembic upgrade head 