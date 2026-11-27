import os
from contextlib import asynccontextmanager
from typing import Annotated, Literal, Optional

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from qdrant_client import QdrantClient

from recommendation_store import Store


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Recommendation(ContractModel):
    title_id: Annotated[str, Field(pattern=r"^TT-\d{4}$")]
    title_name: Annotated[str, Field(min_length=1)]
    primary_genre: Literal["Drama", "Comedy", "Thriller", "Crime", "Documentary", "Sci-Fi",
                           "Fantasy", "Romance", "Horror", "Action", "Animation", "Reality"]
    score: float


class Picks(ContractModel):
    user_id: Annotated[str, Field(pattern=r"^M-\d{5}$")]
    recommendations: Annotated[list[Recommendation], Field(min_length=10, max_length=10)]
    latency_ms: Optional[Annotated[float, Field(ge=0)]] = None


class ErrorResponse(ContractModel):
    user_id: str
    detail: Annotated[str, Field(min_length=1)]


@asynccontextmanager
async def lifespan(app):
    url = os.getenv("QDRANT_URL")
    client = QdrantClient(url=url) if url else QdrantClient(":memory:")
    try:
        app.state.store = Store(client)
        yield
    finally:
        client.close()


app = FastAPI(title="Netflix Top Picks", lifespan=lifespan)


@app.get("/recommendations/{user_id}", response_model=Picks, response_model_exclude_none=True,
         responses={404: {"model": ErrorResponse}})
def recommendations(user_id: str):
    picks = app.state.store.picks(user_id)
    if len(picks) < 10:
        detail = "Member not found" if user_id not in app.state.store.history else "Too few eligible titles"
        error = ErrorResponse(user_id=user_id, detail=detail)
        return JSONResponse(status_code=404, content=error.model_dump())
    return Picks(user_id=user_id, recommendations=picks)
