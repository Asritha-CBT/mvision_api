your_project/
│
├── app/
│   ├── __init__.py
│   ├── main.py
│   │
│   ├── core/
│   │   ├── settings.py
│   │   └── constants.py
│   │
│   ├── db/
│   │   ├── database.py
│   │   └── models.py
│   │
│   ├── routes/
│   │   ├── __init__.py
│   │   └── user_routes.py
│   │
│   ├── services/
│   │   ├── __init__.py
│   │   └── user_service.py
│   │
│   ├── schemas/
│   │   └── user_schema.py
│   │
│   ├── utils/
│   │   └── security.py
│   │
│   └── tests/
│       └── test_users.py
│
├── .env
├── requirements.txt
└── README.md

# python -m venv venv
# venv\Scripts\activate
# pip install -r requirements.txt
# uvicorn mvision.main:app --reload

# pip install --no-build-isolation "git+https://github.com/KaiyangZhou/deep-person-reid.git

# alembic migration
    1. alembic revision --autogenerate -m "migration message"
    2. alembic upgrade head 