from pydantic import BaseModel


class TopKEntry(BaseModel):
    class_name: str
    display: str
    probability: float


class ImageSize(BaseModel):
    width: int
    height: int


class PredictResponse(BaseModel):
    prediction: str
    prediction_display: str
    confidence: float
    confidence_pct: float
    top_k: list[TopKEntry]
    inference_ms: float
    model: str
    image_size: ImageSize


class HealthResponse(BaseModel):
    status: str
    model: str
    checkpoint_age: str
    torch_version: str
    cuda_available: bool


class ClassEntry(BaseModel):
    raw: str
    display: str


class ClassesResponse(BaseModel):
    classes: list[ClassEntry]


class ErrorDetail(BaseModel):
    code: str
    message: str
    detail: str = ""


class ErrorResponse(BaseModel):
    error: ErrorDetail
