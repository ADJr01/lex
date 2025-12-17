class InvalidArgumentError(Exception):
    """
    Raised when Invalid config will be passed in Lex Config.
    """
    def __init__(self,message="Age is outside the valid range"):
        # Call the base class constructor with the required arguments
        super().__init__(message)

        # You can override __str__ to provide a more informative message
        self.message = message

    def __str__(self):
        return self.message
