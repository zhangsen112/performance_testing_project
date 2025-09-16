class ServiceValueError(Exception):
    def __init__(self, value_error_type: str, value_error_field: str):
        self.value_error_type = value_error_type
        self.value_error_field = value_error_field
