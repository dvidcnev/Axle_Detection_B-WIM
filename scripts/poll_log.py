import time
import os
import sys
from datetime import datetime

PATH = os.path.join(os.path.dirname(__file__), '..', 'checkpoints', 'cnn_log.csv')
PATH = os.path.normpath(PATH)

print(f"Starting log poller for: {PATH}")
last_m = None
try:
    while True:
        try:
            if os.path.exists(PATH):
                m = os.path.getmtime(PATH)
                if last_m != m:
                    last_m = m
                    with open(PATH, 'r', encoding='utf-8') as f:
                        lines = f.read().splitlines()
                    if lines:
                        print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), lines[-1])
                    else:
                        print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'log empty')
                    sys.stdout.flush()
            else:
                print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'log missing')
                sys.stdout.flush()
        except Exception as e:
            print('poll error:', e)
            sys.stdout.flush()
        time.sleep(60)
except KeyboardInterrupt:
    print('Poller stopped')
