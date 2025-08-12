import os

import triton
import triton.language as tl

# watchdog_proc.py
import os, sys, signal, subprocess, threading, time

_lock = threading.Lock()
_watch: dict[str, tuple[subprocess.Popen, int]] = {}

def arm(key: str, timeout_s: float):
    ppid = os.getpid()
    rfd, wfd = os.pipe()
    os.set_inheritable(rfd, True)  # pass read end to child

    code = (
        "import os,sys,signal,select;"
        "fd=int(sys.argv[1]); t=float(sys.argv[2]); ppid=int(sys.argv[3]);"
        "r,_,_=select.select([fd],[],[],t);"
        "os._exit(0) if r else os.kill(ppid, signal.SIGKILL)"
    )
    proc = subprocess.Popen(
        [sys.executable, "-c", code, str(rfd), str(timeout_s), str(ppid)],
        close_fds=False,
    )
    os.close(rfd)

    with _lock:
        old = _watch.pop(key, None)
        if old:
            _cancel_unlocked(old)
        _watch[key] = (proc, wfd)

def _cancel_unlocked(entry):
    proc, wfd = entry
    try: os.write(wfd, b"x")
    except OSError: pass
    try: os.close(wfd)
    except OSError: pass
    if proc.poll() is None:
        proc.terminate()
        try: proc.wait(timeout=0.2)
        except Exception: proc.kill()

def disarm(key: str) -> bool:
    with _lock:
        entry = _watch.pop(key, None)
    if not entry:
        return False
    _cancel_unlocked(entry)
    return True


def test_watchdog(device, fresh_triton_cache) -> None:
    @triton.jit
    def kernel_dummy(a: tl.constexpr):
        pass
    def load_start(module, function, name, metadata_group, hash) -> None:
        print("start_hook")
        arm(hash, 2.0)

    def load_end(module, function, name, metadata_group, hash) -> None:
        print("end_hook")
        disarm(hash)

    triton.knobs.runtime.kernel_load_start_hook = load_start
    triton.knobs.runtime.kernel_load_end_hook = load_end
    kernel_dummy[(1, )](1)