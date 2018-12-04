import socket


class Netcat:

    def __init__(self, ip, port):
        self.buff = ""
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((ip, port))

    def read_until(self, data):
        while data not in self.buff:
            self.buff += self.socket.recv(1024)

        pos = self.buff.find(data)
        val = self.buff[:pos + len(data)]
        self.buff = self.buff[pos + len(data):]

        return val

    def write(self, data):
        self.socket.send(data)

    def close(self):
        self.socket.close()