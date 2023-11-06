from pylsl import resolve_stream
from get_lsl_data import PyLSLWrapper, get_all_streams
from pynput import keyboard
import random
import sys
import threading
import os
import time
import numpy as np
import atexit
import argparse
try:
    import matlab.engine
except ImportError:
    print('Matlab engine not found. Please install matlab engine for python')
    sys.exit(0)

def get_rps(data, eng, matlab_pth):
    else:
        # if we don't have a matlab engine, use a python model

"""
Helper functions, don't worry about these!
"""
clear = lambda : os.system('cls' if os.name == 'nt' else 'clear')
pressed_key = None
key_press_event = threading.Event()
wrapper = None

def key_action(key_char):
    global pressed_key
    pressed_key = key_char
    if pressed_key == 'q':
        print('Quitting stuff program')
        key_press_event.set()
        sys.exit(0)
        print('still here...')
    key_press_event.set()

def on_press(key):
    try:
        key_char = key.char
        key_action(key_char)
    except AttributeError:
        pass

def on_release(key):
    # Exit the listener and the while loop when the 'esc' key is pressed
    try:
        key_char = key.char
        if key_char == 'q':
            return False
    except:
       pass

def exit_program():
    print('Finished the gestures!')
    print('set!')
    if wrapper:
        wrapper.end_stream_listener()
    key_press_event.clear()
    listener_thread.stop()
    listener_thread.join()
    sys.exit(0)

atexit.register(exit_program)

if __name__=='__main__':
    # Figure out if we are using matlab engine or python engine
    parser = argparse.ArgumentParser(
        prog='testDataPython',
        description='Run the data testing ')
    parser.add_argument('--matlabmodel', help='Location of the runModel.m file', default='runModel.m')
    args = parser.parse_args()
    if args.matlabmodel:
        eng = matlab.engine.start_matlab()
        # if we have a matlab engine and want to use a matlab model, import it
        pth = os.path.dirname(matlab_pth)
        eng.addpath(pth)
        runModel = lambda data: eng.runModel(data)
    else:
        dir_path = os.path.dirname(os.path.realpath(__file__))
        sys.path.append(dir_path)
        from runPythonModel import get_rps
        runModel = lambda data: get_rps(data)

    # Start the key listener thread
    listener_thread = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener_thread.start()
    clear()
    recv = None

    # Pick the marker stream to tell the program when to collect data
    marker_streams = get_all_streams()
    for stream in marker_streams:
        if stream.name() == 'marker_send':
            recv = stream
            inferred_markers = StreamInfo('inferred_markers', 'markers', 1, 0, 'inferred_markers')
            inferred_out = StreamOutlet(inferred_markers)
    if not recv:
        print('No marker stream found. Make sure there is a computer with a marker stream running')
        sys.exit(0)
    clear() 

    # Find the wristband stream
    stream_name = None
    while not stream_name:
        key_press_event.clear()
        streams = resolve_stream()
        for i, stream in enumerate(streams):
            print(f"\033[1m{i}: {stream.name()}\033[0m")
        print('Which stream is your wristband? (r) to reload list of available/ (q) to quit')
        key_press_event.wait()
        if pressed_key:
            pressed = pressed_key.strip()
            if pressed == 'r':
                print('Reloading')
                continue
            elif pressed == 'q':
                sys.exit(0)
            else:
                try:
                    stream_name = streams[int(pressed_key)].name()
                    if stream_name == 'marker_send':
                        print('Please choose a different stream')
                        stream_name = None
                        continue
                    clear()
                except:
                    print('Please enter a valid int listed')

    # Start the stream listener and data collection
    wrapper = PyLSLWrapper(stream_name)
    wrapper.launch_stream_listener()
    while True:
        marker, timestamp = recv.pull_sample()
        if marker == 1:
            tstamp = time.time()
            print('3\r')
            time.sleep(1)
            print('2\r')
            time.sleep(1)
            print('1\r')
            time.sleep(.5)
            tstamp_start = wrapper.get_curr_timestamp
            time.sleep(.5)
            print('Shoot!')
            time.sleep(2)
            data = wrapper.get_data_from(tstamp_start)
            inference = runModel(data)
            inferred_out.push_sample([inference])
        elif marker==99:
            sys.exit(0)
