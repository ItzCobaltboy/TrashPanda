import os
from datetime import datetime
import yaml
from colorama import init, Fore, Style

# Initialize colorama
init(autoreset=True)

# Load config
def load_config():
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

config = load_config()
log_dir = config['logging']['logs_dir']
os.makedirs(log_dir, exist_ok=True)

log_file = os.path.join(
    log_dir, datetime.now().strftime("log_%H-%M-%S_%d-%m.txt")
)

print("Log file path:", log_file)


with open(log_file, "w") as f:
    f.write("Log file created on: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n\n")

# Core log function
def _write_log(level, message, color):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted = f"{timestamp} - {level.upper()} - {message}"
    
    # Append to log file
    with open(log_file, "a") as f:
        f.write(formatted + "\n")

    # Print to terminal with color
    print(color + f"[{level.upper()}]" + Style.RESET_ALL + f" {message}")
    
    

# Public functions
def log_info(message):
    _write_log("info", message, Fore.CYAN)

def log_warning(message):
    _write_log("warning", message, Fore.YELLOW)

def log_error(message):
    _write_log("error", message, Fore.RED)

def log_debug(message):
    _write_log("debug", message, Fore.GREEN)

