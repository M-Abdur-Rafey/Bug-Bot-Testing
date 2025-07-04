#!/bin/bash

pm2 stop translator-extension-server
pm2 delete translator-extension-server
pm2 flush translator-extension-server
pm2 start "uvicorn main:app --port 7001 --workers 5" --name translator-extension-server