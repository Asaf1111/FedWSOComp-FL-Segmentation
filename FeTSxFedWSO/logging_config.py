import logging
import os

# Ensure the logs directory exists
os.makedirs("logs", exist_ok=True)

# Client logger (used only if not passed from client_app.py)
client_logger = logging.getLogger("client_logger")
client_logger.setLevel(logging.DEBUG)
if not client_logger.hasHandlers():
    ch = logging.FileHandler("logs/client.log")
    ch.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s"))
    client_logger.addHandler(ch)

# Server logger — leave empty, actual log file is set in server_app.py
server_logger = logging.getLogger("server_logger")
server_logger.setLevel(logging.DEBUG)
#  Do NOT assign FileHandler here anymore

# Clean up default log files (optional legacy)
log_files = ["logs/client.log", "logs/server.log"]
for file in log_files:
    open(file, 'w').close()
    print(f"Cleared: {file}")
