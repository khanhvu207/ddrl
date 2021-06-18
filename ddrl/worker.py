import socket
import concurrent.futures 

class Worker:
    def __init__(self):
        self.executor = concurrent.futures.ThreadPoolExecutor()

        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._connect_to_server()
        self.executor.submit(self._listen_to_server)
        self.executor.submit(self._send_msg_to_server)

    def _connect_to_server(self, ip='127.0.0.1', port=23333):
        self.s.connect((ip, port))
    
    def _listen_to_server(self):
        while True:
            msg = self.s.recv(4096)
            if len(msg):
                msg = msg.decode('utf-8')
                print(f'>SERVER: {msg}')
    
    def _send_msg_to_server(self):
        while True:
            rawinput = input()
            if len(rawinput):
                self.s.sendall(rawinput.encode('utf-8'))
            
    