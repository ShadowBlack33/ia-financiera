import os
from pathlib import Path

def cleanup_logs(log_dir: Path = Path("logs"), keep: int = 5):
    if not log_dir.exists():
        return

    logs = sorted([f for f in log_dir.glob("*.log")], key=os.path.getmtime, reverse=True)
    if len(logs) <= keep:
        return

    to_delete = logs[keep:]
    for f in to_delete:
        try:
            f.unlink()
            print(f"🧹 Log eliminado: {f.name}")
        except Exception as e:
            print(f"❌ Error eliminando {f.name}: {e}")
