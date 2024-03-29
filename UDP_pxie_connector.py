import socket

class UDP_sender():
    
    def __init__(self, IpAddress, Port):
        
        self.ipaddress  = IpAddress
        self.port  = Port
        self.serverAddressPort = (IpAddress, Port)
        self.UDPClientSocket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
    
    def send_message(self, text):
        bytesToSend = str.encode(text)
        self.UDPClientSocket.sendto(bytesToSend, self.serverAddressPort)
        
        