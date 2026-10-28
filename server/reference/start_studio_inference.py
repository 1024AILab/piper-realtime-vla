"""Idempotently start the already installed LoRA server; never touches training."""
import fcntl
import json
from pathlib import Path
import socket
import subprocess
import sys
import time

ROOT = Path('/root/autodl-tmp/piper-training')
RUN = ROOT / 'runtime/studio-inference'
RUN.mkdir(parents=True, exist_ok=True)


def listening():
    with socket.socket() as s:
        s.settimeout(1)
        return s.connect_ex(('127.0.0.1', 8000)) == 0


def main():
    with (RUN / 'launch.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if listening():
            print('模型服务已在 127.0.0.1:8000 运行；客户端将核验模型元数据。')
            return
        pidfile = RUN / 'pid'
        live = False
        if pidfile.exists():
            pid = int(pidfile.read_text())
            cmd = Path(f'/proc/{pid}/cmdline')
            live = cmd.exists() and b'serve_piper_studio.py' in cmd.read_bytes()
        if not live:
            free = int(subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True).splitlines()[0])
            if free < 18000:
                raise RuntimeError(f'当前剩余显存 {free} MiB，低于此服务并行启动阈值 18000 MiB')
            with (RUN / 'server.log').open('a') as log:
                proc = subprocess.Popen([sys.executable, '-u', str(ROOT / 'serve_piper_studio.py')],
                    stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
            pidfile.write_text(str(proc.pid))
            print(f'模型正在加载与预热，进程 {proc.pid}…', flush=True)
        for _ in range(90):
            if listening():
                print('LoRA 推理服务已就绪。')
                return
            time.sleep(1)
        raise RuntimeError(f'服务尚未就绪，请检查 {RUN / "server.log"}')


if __name__ == '__main__':
    main()
