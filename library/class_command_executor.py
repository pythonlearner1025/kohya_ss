import subprocess
import psutil
from library.custom_logging import setup_logging
import threading

# Set up logging
log = setup_logging()


class CommandExecutor:
    def __init__(self):
        self.process = None
    def execute_command(self, run_cmd):
            if self.process and self.process.poll() is None:
                log.info('The command is already running. Please wait for it to finish.')
            else:
                # Using subprocess.PIPE allows capturing the stdout and stderr
                self.process = subprocess.Popen(run_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

                # Start threads to read stdout and stderr without blocking
                threading.Thread(target=self._read_stream, args=(self.process.stdout,)).start()
                threading.Thread(target=self._read_stream, args=(self.process.stderr,)).start()

    def _read_stream(self, stream):
        for line in stream:
            print(line, end='')  # Print each line from the stream to stdout
        stream.close()

    def kill_command(self):
        if self.process and self.process.poll() is None:
            try:
                parent = psutil.Process(self.process.pid)
                for child in parent.children(recursive=True):
                    child.kill()
                parent.kill()
                log.info('The running process has been terminated.')
            except psutil.NoSuchProcess:
                log.info('The process does not exist.')
            except Exception as e:
                log.info(f'Error when terminating process: {e}')
        else:
            log.info('There is no running process to kill.')
