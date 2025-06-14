import os
import time
import tempfile
import contextlib
import multiprocessing
import signal
import subprocess
from contextlib import redirect_stdout, redirect_stderr, contextmanager
import faulthandler
import io
import numpy as np
from multiprocessing import Array, Value, Manager
import sys
import types
import unittest
from typing import Tuple

PASS = "pass"
FAIL = "fail"
TIMEOUT = "timeout"

_SUCCESS = 0
_FAILED = 1
_TIMEOUT = 2
_UNKNOWN = 3

TIMEOUT_LIMIT = 240.0

_mapping = {_SUCCESS: PASS, _FAILED: FAIL, _TIMEOUT: TIMEOUT, _UNKNOWN: None}


def unsafe_execute(
    code: str,
    test_code: str,
    timeout: float,
    stat,  # Value
    details,  # Array
):
    with safe_environment(), create_tempdir():
        # These system calls are needed when cleaning up tempdir.
        import os
        import shutil
        import builtins

        rmtree = shutil.rmtree
        rmdir = os.rmdir
        chdir = os.chdir
        # Disable functionalities that can make destructive changes to the test.
        reliability_guard()
        module_name = "__test__"
        new_module = types.ModuleType(module_name)
        # Set necessary attributes for the module
        new_module.__dict__.update(
            {
                "__builtins__": builtins,
                "__file__": f"{module_name}.py",
                "__package__": None,
                "__doc__": None,
                "sys": sys,
                "os": os,
                "environ": os.environ,
            }
        )

        try:
            full_code = code + "\n" + test_code

            with swallow_io():
                exec(compile(full_code, f"{module_name}.py", "exec"), new_module.__dict__)
                sys.modules[module_name] = new_module
                TestCases = getattr(new_module, "TestCases")
                loader = unittest.TestLoader()
                suite = loader.loadTestsFromTestCase(TestCases)
                test_result = unittest.TestResult()
                start_time = time.time()
                with time_limit(timeout):
                    suite.run(test_result)

            issues = test_result.failures + test_result.errors
            for test, trace in issues:
                details[test.id().split(".")[-1]] = trace
            stat.value = _SUCCESS
        except BaseException as e:
            details["ALL"] = str(e)
            stat.value = _FAILED
        # Needed for cleaning up.
        shutil.rmtree = rmtree
        os.rmdir = rmdir
        os.chdir = chdir


def untrusted_check(code: str, test_code: str, min_time_limit: float = 10, gt_time_limit: float = 60) -> Tuple[str, np.ndarray]:
    min_time_limit = max(min_time_limit, gt_time_limit)
    timeout = max(os.getenv("BIGCODEBENCH_TIMEOUT_PER_TASK", TIMEOUT_LIMIT), min_time_limit) + 1
    # shared memory objects
    stat = Value("i", _UNKNOWN)
    manager = Manager()
    details = manager.dict()

    p = multiprocessing.Process(
        target=unsafe_execute,
        args=(
            code,
            test_code,
            timeout,
            stat,
            details,
        ),
    )
    p.start()
    p.join(timeout=timeout + 1)
    if p.is_alive():
        p.terminate()
        time.sleep(0.1)
    if p.is_alive():
        p.kill()
        time.sleep(0.1)

    stat = _mapping[stat.value]
    # convert details to a dict
    details = dict(details)

    if not stat:
        stat = TIMEOUT
    if stat == PASS:
        if details:
            stat = FAIL

    return stat, details


@contextlib.contextmanager
def safe_environment():
    # Save original functions
    original_kill = os.kill
    original_killpg = os.killpg
    original_system = os.system
    original_subprocess_call = subprocess.call
    original_subprocess_check_output = subprocess.check_output
    original_subprocess_run = subprocess.run
    original_subprocess_popen = subprocess.Popen
    original_os_popen = os.popen
    original_os_execv = os.execv
    original_os_execvp = os.execvp
    original_os_execvpe = os.execvpe

    current_pid = os.getpid()
    current_pgid = os.getpgid(current_pid)
    manager = multiprocessing.Manager()
    child_pids = manager.list()

    def safe_kill(pid, sig):
        try:
            pgid = os.getpgid(pid)
            if pid == current_pid or pid in child_pids:
                original_kill(pid, sig)
            else:
                print(f"Prevented attempt to kill PID {pid} with signal {sig}")
        except ProcessLookupError:
            pass

    def safe_killpg(pgid, sig):
        if pgid == current_pgid or pgid in {os.getpgid(pid) for pid in child_pids}:
            original_killpg(pgid, sig)
        else:
            print(f"Prevented attempt to kill PGID {pgid} with signal {sig}")

    def safe_system(command):
        print(f"Intercepted system command: {command}")
        if "kill" in command or "killall" in command:
            return 0  # Simulate successful execution without doing anything
        return original_system(command)

    def safe_subprocess_call(command, *args, **kwargs):
        print(f"Intercepted subprocess call: {command}")
        if "kill" in command or "killall" in command:
            return 0  # Simulate successful execution without doing anything
        return original_subprocess_call(command, *args, **kwargs)

    def safe_subprocess_check_output(command, *args, **kwargs):
        print(f"Intercepted command: {command}")
        if "ps" in command:
            return b""  # Simulate no processes found
        return original_subprocess_check_output(command, *args, **kwargs)

    def safe_subprocess_run(*args, **kwargs):
        print(f"Intercepted subprocess run command: {args}")
        if "kill" in args[0] or "killall" in args[0]:
            return subprocess.CompletedProcess(args, 0, b"", b"")  # Simulate successful execution
        return original_subprocess_run(*args, **kwargs)

    class SafePopen(subprocess.Popen):
        def __init__(self, *args, **kwargs):
            print(f"Intercepted Popen command: {args}")
            kwargs["preexec_fn"] = os.setsid  # Start the process in a new session
            super().__init__(*args, **kwargs)
            child_pids.append(self.pid)

        def communicate(self, *args, **kwargs):
            try:
                return super().communicate(*args, **kwargs)
            except subprocess.TimeoutExpired:
                print("Timeout expired, intercepted and returning None")
                return None, None

        def kill(self):
            print(f"Intercepted kill call for PID {self.pid}")
            safe_kill(self.pid, signal.SIGTERM)

        def terminate(self):
            print(f"Intercepted terminate call for PID {self.pid}")
            safe_kill(self.pid, signal.SIGTERM)

    def safe_os_popen(command):
        print(f"Intercepted os.popen command: {command}")
        if "kill" in command or "killall" in command:
            return os.popen("echo Intercepted")
        return original_os_popen(command)

    def safe_exec(*args, **kwargs):
        print(f"Intercepted exec command: {args}")

    # Override the risky functions with the safe versions
    os.kill = safe_kill
    os.killpg = safe_killpg
    os.system = safe_system
    subprocess.call = safe_subprocess_call
    subprocess.check_output = safe_subprocess_check_output
    subprocess.run = safe_subprocess_run
    subprocess.Popen = SafePopen
    os.popen = safe_os_popen
    os.execv = safe_exec
    os.execvp = safe_exec
    os.execvpe = safe_exec

    try:
        yield
    finally:
        for pid in child_pids:
            try:
                os.kill(pid, signal.SIGTERM)
                for _ in range(10):
                    time.sleep(0.1)
                    try:
                        os.kill(pid, 0)
                    except ProcessLookupError:
                        break
                else:
                    os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except Exception as e:
                print(f"Error handling process {pid}: {e}")

        os.kill = original_kill
        os.killpg = original_killpg
        os.system = original_system
        subprocess.call = original_subprocess_call
        subprocess.check_output = original_subprocess_check_output
        subprocess.run = original_subprocess_run
        subprocess.Popen = original_subprocess_popen
        os.popen = original_os_popen
        os.execv = original_os_execv
        os.execvp = original_os_execvp
        os.execvpe = original_os_execvpe


@contextlib.contextmanager
def chdir(root):
    if root == ".":
        yield
        return
    cwd = os.getcwd()
    os.chdir(root)
    try:
        yield
    except BaseException as exc:
        raise exc
    finally:
        os.chdir(cwd)


@contextlib.contextmanager
def create_tempdir():
    with tempfile.TemporaryDirectory() as dirname:
        with chdir(dirname):
            yield dirname


class redirect_stdin(redirect_stdout):
    _stream = "stdin"


# Utility Classes
class TimeoutException(Exception):
    pass


class WriteOnlyStringIO(io.StringIO):
    """StringIO that only supports writing"""

    def read(self, *args, **kwargs):
        raise io.UnsupportedOperation("not readable")

    def readline(self, *args, **kwargs):
        raise io.UnsupportedOperation("not readable")

    def readlines(self, *args, **kwargs):
        raise io.UnsupportedOperation("not readable")


# Context Managers
@contextmanager
def swallow_io():
    """Suppress all I/O during code execution"""
    stream = WriteOnlyStringIO()
    with redirect_stdout(stream):
        with redirect_stderr(stream):
            with redirect_stdin(stream):
                yield


@contextmanager
def time_limit(seconds: float):
    """Set a time limit for code execution"""

    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")

    signal.setitimer(signal.ITIMER_REAL, seconds)
    signal.signal(signal.SIGALRM, signal_handler)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


def reliability_guard():
    import builtins

    builtins.exit = None
    builtins.quit = None

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["TZ"] = "UTC"
    faulthandler.disable()

    import matplotlib.pyplot as plt

    plt.close("all")
