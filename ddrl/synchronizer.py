import pickle
import threading


class Synchronizer:
    def __init__(self, config, agent):
        self.config = config
        self.agent = agent
        self.data_string = None

        self._lock = threading.Lock()

    def update_weights(self):
        weight = self.agent.get_weights()
        with self._lock:
            weight_dict = {"net": weight}
            self.data_string = pickle.dumps(weight_dict)

    def send_weights(self, client):
        with self._lock:
            msg = bytes(f"{len(self.data_string):<{15}}", "utf-8") + self.data_string
        client.sendall(msg)
        print("Weights sent!")
