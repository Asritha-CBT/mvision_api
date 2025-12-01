your_project/
│
├── app/
│   ├── __init__.py
│   ├── main.py
│   │
│   ├── core/
│   │   ├── config.py
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


uvicorn mvision.main:app --reload
