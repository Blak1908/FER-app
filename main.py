from fastapi import FastAPI
from app.api.fer import router


app = FastAPI(openapi_url="/api/v1/fer/openapi.json", docs_url="/api/v1/fer/docs")
app.include_router(router, prefix='/api/v1/fer', tags=['fer'])