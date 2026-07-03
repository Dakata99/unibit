import logging

class LevelColorFormatter(logging.Formatter):
    COLORS = {
        "DEBUG":    "\033[36m",  # cyan
        "INFO":     "\033[34m",  # blue
        "WARNING":  "\033[33m",  # yellow
        "ERROR":    "\033[31m",  # red
        "CRITICAL": "\033[41m",  # red background
    }
    RESET = "\033[0m"

    def format(self, record):
        level = record.levelname
        color = self.COLORS.get(level, self.RESET)
        record.levelname = f"{color}[{level}]{self.RESET}"
        return super().format(record)


# Create a global logger instance
log = logging.getLogger("pipeline")
log.setLevel(logging.DEBUG)

# Avoid adding handlers multiple times if imported repeatedly
if not log.handlers:
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(LevelColorFormatter("%(levelname)s %(message)s"))
    log.addHandler(handler)
