import socket
import threading

from serverFunc import *

# Configure server 
host = 'localhost'
port = 8888

# Create server socket 
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM) # IPv4/TCP
server_socket.bind((host, port))
server_socket.listen(5)     # Up to 5 connections in the queue

print('Server running. Waiting for connections...')

while True: 
    # Accept new connection 
    client_socket, client_address = server_socket.accept()
    print("Connected client: ", client_address)

    # Manage the client
    thread = threading.Thread(target=manage_client, args=(client_socket,))
    thread.start()