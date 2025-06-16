import sys

def error_message_detail(error, error_detail):
    """
    This function takes an error and its details as input and returns a formatted string with the error message.
    """
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = f"Error occured in script name [{file_name}] at line number [{exc_tb.tb_lineno}] error message [{str(error)}]"
    return error_message

class CustomException(Exception):
    """
    This class is a custom exception that inherits from the built-in Exception class.
    It takes an error and its details as input and returns a formatted string with the error message.
    """
    def __init__(self, error_message, error_detail):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail=error_detail)

    def __str__(self):
        return self.error_message
    
    