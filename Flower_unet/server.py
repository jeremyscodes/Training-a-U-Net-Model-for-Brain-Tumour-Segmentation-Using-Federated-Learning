import flwr as fl
import socket


def get_ip_address():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # Doesn't need to be reachable, the OS just uses this to determine the most
        # appropriate network interface to use.
        s.connect(('10.254.254.254', 1))
        IP = s.getsockname()[0]
    except Exception:
        IP = '127.0.0.1'
    finally:
        s.close()
    return IP

server_ip = get_ip_address()
with open("server_ip.txt", "w") as f:
    f.write(server_ip)

fl.server.start_server(
    server_address=server_ip+":8080",
    config=fl.server.ServerConfig(num_rounds=3),
)
