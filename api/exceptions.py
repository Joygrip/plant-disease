from fastapi import Request
from fastapi.responses import JSONResponse


class PlantAIException(Exception):
    def __init__(self, code: str, message: str, status_code: int, detail: str = ""):
        self.code = code
        self.message = message
        self.status_code = status_code
        self.detail = detail
        super().__init__(message)


class InvalidImageError(PlantAIException):
    def __init__(self, detail: str = ""):
        super().__init__("INVALID_IMAGE", "File is not a valid image", 400, detail)


class FileTooLargeError(PlantAIException):
    def __init__(self, max_mb: int):
        super().__init__("FILE_TOO_LARGE", f"File exceeds {max_mb} MB limit", 413)


class InvalidContentTypeError(PlantAIException):
    def __init__(self, content_type: str):
        super().__init__(
            "INVALID_CONTENT_TYPE",
            "Content-Type must be an image",
            415,
            f"Got: {content_type}",
        )


class ImageTooSmallError(PlantAIException):
    def __init__(self, width: int, height: int):
        super().__init__(
            "IMAGE_TOO_SMALL",
            "Image is too small (minimum 50\u00d750 px)",
            422,
            f"Got: {width}\u00d7{height}",
        )


class ModelNotLoadedError(PlantAIException):
    def __init__(self):
        super().__init__("MODEL_NOT_LOADED", "Model is not loaded yet", 503)


class UnknownModelError(PlantAIException):
    def __init__(self, model_name: str):
        super().__init__("UNKNOWN_MODEL", f"Unknown model '{model_name}'", 400)


class InternalError(PlantAIException):
    def __init__(self, detail: str = ""):
        super().__init__("INTERNAL_ERROR", "An internal error occurred", 500, detail)


async def plant_ai_exception_handler(request: Request, exc: PlantAIException) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": {"code": exc.code, "message": exc.message, "detail": exc.detail}},
    )
