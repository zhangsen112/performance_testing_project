from fastapi.responses import JSONResponse


def get_ok_response(data: dict | list, message: str | None = ""):
    response_data = {"success": True, "data": data, "message": message}
    return JSONResponse(status_code=200, content=response_data)


def get_error_response(message: str | None = ""):
    response_data = {"success": False, "data": None, "message": message}
    return JSONResponse(status_code=200, content=response_data)
