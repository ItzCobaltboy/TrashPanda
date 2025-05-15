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
debug_mode = config['logging']['debug']
os.makedirs(log_dir, exist_ok=True)

log_file = os.path.join(
    log_dir, datetime.now().strftime("log_%H-%M-%S_%d-%m.txt")
)



with open(log_file, "w") as f:
    f.write("Log file created on: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n\n")

def write_log(user ,level, message, color):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted = f"{timestamp} -{user} - {level.upper()} - {message}"
    
    # Append to log file
    with open(log_file, "a") as f:
        f.write(formatted + "\n")

    # Print to terminal with color
    print(color + f"[{level.upper()}]"+ Fore.MAGENTA + f"[{user}]" + Style.RESET_ALL + f" {message}")
    
write_log("Logger", "info", f"Log file path: {log_file}", Fore.CYAN)

if debug_mode:
    write_log("Logger", "debug", f"DEBUGGING MODE ACTIVE", Fore.GREEN)

class logger():

    def __init__(self):
        self.user = "Logger"

    def log_info(self, message):
        write_log(self.user, "info", message, Fore.CYAN)

    def log_warning(self, message):
        write_log(self.user, "warning", message, Fore.YELLOW)

    def log_error(self, message):
        write_log(self.user, "error", message, Fore.RED)

    def log_debug(self, message):
        if debug_mode:
            write_log(self.user, "debug", message, Fore.GREEN)
        else:
            return