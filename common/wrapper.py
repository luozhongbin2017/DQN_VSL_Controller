'''The wrapper helps to wrap the environment into preprocessed data.'''

import numpy as np
from collections import deque
from skimage import transform

def preprocess_frame(frame):
    
    # Crop the screen (remove part that contains no information)
    # [Up: Down, Left: right]
    """
    preprocess_frame:
    Take a frame.
    Resize it.
        __________________
        |                 |
        |                 |
        |                 |
        |                 |
        |_________________|
        
        to
        _____________
        |            |
        |            |
        |            |
        |____________|
    Normalize it.
    
    return preprocessed_frame
    
    """
    LEFT, RIGHT = -110, 689
    UP, DOWN = 57, 38
    cropped_frame = frame[UP: DOWN, LEFT: RIGHT]
    
    # Normalize Pixel Values
    normalized_frame = cropped_frame/255.0
    
    # Resize
    preprocessed_frame = transform.resize(normalized_frame, [20,880])
    
    return preprocessed_frame # 20x880x1 frame

# Stack images for giving have a sense of motion to our Neural Network.
    
stack_size = 3 # We stack 3 frames

# Initialize deque with zero-images one array for each image
stacked_frames  =  deque([np.zeros((100,120), dtype=np.int) for i in range(stack_size)], maxlen = 3) 

def stack_frames(stacked_frames, state, is_new_episode):  #Abandoned
    # Preprocess frame
    frame = preprocess_frame(state)
    
    if is_new_episode:
        # Clear our stacked_frames
        stacked_frames = deque([np.zeros((100,120), dtype=np.int) for i in range(stack_size)], maxlen = 3)
        
        # Because we're in a new episode, copy the same frame 4x
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        
        # Stack the frames
        stacked_state = np.stack(stacked_frames, axis=2)

    else:
        # Append frame to deque, automatically removes the oldest frame
        stacked_frames.append(frame)

        # Build the stacked state (first dimension specifies different frames)
        stacked_state = np.stack(stacked_frames, axis=2) 
    
    return stacked_state, stacked_frames