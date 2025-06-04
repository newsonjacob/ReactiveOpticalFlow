import os

DEBUG_LOGGING = os.environ.get("DEBUG_LOGGING", "0") == "1"

def debug_print(*args, **kwargs):
    if DEBUG_LOGGING:
        print(*args, **kwargs)
