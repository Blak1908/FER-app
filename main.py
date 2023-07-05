from fastapi import FastAPI
from app.api.s2t import router


app = FastAPI(openapi_url="/api/v1/s2t/openapi.json", docs_url="/api/v1/s2t/docs")
app.include_router(router, prefix='/api/v1/s2t', tags=['s2t'])



# uvicorn main:app --reload