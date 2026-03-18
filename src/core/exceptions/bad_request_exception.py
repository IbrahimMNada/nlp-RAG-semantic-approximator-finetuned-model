class BadRequestException(Exception):
    def __init__(self, response: dict):
        self.response = response



