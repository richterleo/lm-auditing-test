from datetime import datetime

def create_run_string():
    # Get current date and time
    current_datetime = datetime.now()
    # Format the datetime to a string
    datetime_str = current_datetime.strftime("%Y-%m-%d_%H:%M:%S")
    # Create and return the string with "run" appended with the current date and time
    return f"run_{datetime_str}"