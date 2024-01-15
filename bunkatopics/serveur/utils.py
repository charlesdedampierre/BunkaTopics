import psutil


def is_server_running():
    for process in psutil.process_iter(attrs=["pid", "name"]):
        try:
            if "node" in process.info["name"].lower():
                connections = process.connections()
                for conn in connections:
                    if conn.laddr.port == 3000:
                        return True
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    return False


def kill_server():
    for process in psutil.process_iter(attrs=["pid", "name"]):
        try:
            if "node" in process.info["name"].lower():
                connections = process.connections()
                for conn in connections:
                    if conn.laddr.port == 3000:
                        process.terminate()
                        return
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
