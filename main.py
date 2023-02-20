from fastapi import FastAPI



app = FastAPI(openapi_url="/api/v1/fer/openapi.json", docs_url="/api/v1/fer/docs")
