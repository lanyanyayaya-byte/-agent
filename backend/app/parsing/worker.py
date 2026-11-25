"""Own a local parser subprocess through both graceful stop and parent death."""

import os
import subprocess
from concurrent.futures import CancelledError
from contextlib import contextmanager
from threading import Event, Thread


def check_cancelled(cancelled: Event | None) -> None:
    if cancelled is not None and cancelled.is_set():
        raise CancelledError()


@contextmanager
def open_worker(command: list[str], *, cwd, cancelled: Event | None):
    check_cancelled(cancelled)
    read_fd, write_fd = os.pipe()
    try:
        try:
            worker = subprocess.Popen(
                [*command, str(read_fd)], cwd=cwd, pass_fds=(read_fd,),
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            )
        finally:
            os.close(read_fd)
        with worker:
            finished = Event()

            def finish_worker() -> None:
                nonlocal write_fd
                if write_fd is None:
                    return
                # Closing the pipe also stops a parser running under a frozen onefile launcher.
                os.close(write_fd)
                write_fd = None
                try:
                    worker.wait(timeout=0.5)
                except subprocess.TimeoutExpired:
                    worker.kill()

            def stop_on_cancel() -> None:
                while not finished.wait(0.05):
                    if cancelled.is_set():
                        finish_worker()
                        return

            monitor = None
            if cancelled is not None:
                monitor = Thread(target=stop_on_cancel, name="parser-cancellation")
                monitor.start()
            try:
                yield worker
            finally:
                finished.set()
                if monitor is not None:
                    monitor.join()
                finish_worker()
    finally:
        if write_fd is not None:
            os.close(write_fd)


def watch_owner(fd: int) -> None:
    def exit_on_disconnect() -> None:
        try:
            os.read(fd, 1)
        finally:
            os._exit(1)

    Thread(target=exit_on_disconnect, name="parser-owner", daemon=True).start()


def communicate(worker: subprocess.Popen, data: bytes, timeout: float, cancelled: Event | None) -> bytes:
    result, _ = worker.communicate(data, timeout=timeout)
    check_cancelled(cancelled)
    return result
