#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
sys.path.append("../")
from websocket_server import WebsocketServer
import json
from pose_detection import pose_detection
def new_client(client, server):
    server.send_message_to_all("Hey all, a new client has joined us")


def message_received(client, server, message):
    try:
        result =pose_detection(message)[0].tolist()
        server.send_message(client, json.dumps(result))
    except Exception as e:
        print(e)


# Called for every client disconnecting
def client_left(client, server):
    print("Client(%d) disconnected" % client['id'])


server = WebsocketServer(8001, host='localhost')

print("Server start, print Ctrl + C to quit")

server.set_fn_new_client(new_client)
server.set_fn_message_received(message_received)
server.set_fn_client_left(client_left)

server.run_forever()
