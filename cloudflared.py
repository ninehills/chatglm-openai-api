#!/usr/bin/env python
# coding=utf-8
import atexit
import subprocess

from threading import Timer


def start_cloudflared(command, port):
    cloudflared = subprocess.Popen(
        [command, 'tunnel', '--url', 'http://127.0.0.1:' + str(port)],
        stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    atexit.register(cloudflared.terminate)


def run(command, port):
    # Starting the Cloudflared tunnel in a separate thread.
    thread = Timer(2, start_cloudflared, args=(command, port,))
    thread.setDaemon(True)
    thread.start()
