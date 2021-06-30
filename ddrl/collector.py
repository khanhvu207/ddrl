import pickle
import concurrent.futures

from .synchronizer import Synchronizer


class Collector:
    def __init__(self, config, buffer, synchronizer):
        self.config = config
        self.buffer = buffer
        self.synchronizer = synchronizer

        self.msg_buffer_len = 65536
        self.msg_length_padding = 15

        self.executor = concurrent.futures.ThreadPoolExecutor()

    def got_new_worker(self, client, address):
        self.executor.submit(self._worker_handler, client, address)

    def _worker_handler(self, client, address):
        client_ip, client_port = address
        new_msg = True
        data = b""
        counter = 0

        while True:
            msg = client.recv(self.msg_buffer_len)
            if len(msg):
                if new_msg:
                    msg_len = int(msg[: self.msg_length_padding])
                    msg = msg[self.msg_length_padding :]
                    new_msg = False
                data += msg
                if len(data) == msg_len:
                    batch = pickle.loads(data)
                    self.buffer.add(batch)
                    counter += 1
                    if counter % self.config["worker"]["sync_every"] == 0:
                        self.synchronizer.send_weights(client)
                    new_msg = True
                    data = b""
        client.close()
