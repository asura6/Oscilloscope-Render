import os
import numpy as np
import glob
import imageio
from scipy.spatial import distance
from tsp_solver.greedy import solve_tsp # https://github.com/dmishin/tsp-solver
from scipy.io.wavfile import write

"""
Convert RGBA array to graph coordinates, i.e. a list of coordinates for 
the pixels that are turned ON. All pixels with values above the threshold
will be turned ON
"""
def convert_RGBA_to_coords(RGBA_arr, threshold=127):
    # Discard alpha channel
    data = np.delete(RGBA_arr, -1, axis=-1)
    # Collapse RGB values to a single value using a average-filter
    data = np.mean(data, axis=-1)
    # Apply threshold to array to set pixels to ON or OFF (1 or 0)
    on_coordinates = np.transpose((data > threshold).nonzero())
    return on_coordinates

"""
Get an array with Euclidean distances between each coordinates in a 2d-array of coordinates
"""
def _get_distance_matrix_(positions):
    dist_matrix = distance.cdist(positions, positions)
    return dist_matrix

"""
Convert a list of 2D-points to an efficient path that passes through all points
"""
def get_path(positions):
    D = _get_distance_matrix_(on_coordinates)
    path = solve_tsp(D)
    return path

if __name__ == '__main__':
    # Define location of animation
    animation_src_dir = '/mnt/c/tmp/'
    animation_src_paths = glob.glob(os.path.join(animation_src_dir, '*.png'))

    # Wave setup
    sampleRate = 44100 # Standard audio sampling
    maxVal = 255 # Max brightness value in images
    pointRepeat = 1 # Lower should give smoother image
    frameLength = 3/60 # multiple of 1 s

    ch1, ch2 = [], []
    for n, im_file in enumerate(animation_src_paths):        
        print(f"On image {im_file}")
        # Get the image for the current frame
        im = imageio.imread(im_file)

        # Solve the traveling salesman problem for the image
        on_coordinates = convert_RGBA_to_coords(im)
        path = get_path(on_coordinates)

        # Encode the paths as audio
        for t in range(int((sampleRate * frameLength) / pointRepeat)):
            for n in range(pointRepeat):
                try:
                    x, y = on_coordinates[path[t % len(path)]]
                except:
                    x, y = 32, 32 # If no path then just hang out at the middle of screen
                ch1.append(x / maxVal)
                ch2.append(y / maxVal)

        # The result is saved in a stereo .wav-file
        write('ring_animation.wav', sampleRate, np.array([ch1,ch2]).transpose())
