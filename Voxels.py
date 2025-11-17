import numpy as np
import matplotlib.pyplot as plt
from perlin_noise import PerlinNoise
import math


def noise_value(noise_generator, i, j):
    value = 0
    amplitude = 1
    frequency = 1
    for k in range(octaves):
        value += amplitude * noise_generator[k]([i * frequency / scale, j * frequency / scale])
        amplitude *= persistence
        frequency *= lacunarity

    return value


def noise_array():
    noise_generator=[]
    for i in range(octaves):
        noise_generator.append(PerlinNoise(octaves = 1))

    array = np.zeros((rows, columns))
    for i in range(rows):
        for j in range(columns):
            array[i][j] = noise_value(noise_generator, i, j)
  
    return array


def normalize():
    maxi = np.max(noise)
    mini = np.min(noise)

    for i in range(rows):
        for j in range(columns):
            noise[i][j] = np.round(((noise[i][j] - mini + factor) / (maxi - mini)) * depth).astype(int)

            if noise[i][j] < 1:
                noise[i][j] = 1
            
            elif noise[i][j] > depth:
                noise[i][j] = depth

    return noise


def colourmap(): 
    colour_noise = np.zeros((rows, columns, depth, 3))
                        
    for i in range(rows):
        for j in range(columns):
            for k in range(depth):
                if voxels[i, j, k]:
                    noise_unormalized = noise[i][j] / depth

                    if k < 0.3 * depth: # Sea
                        colour_noise[i][j][k] = [0, 0, int(255 * noise_unormalized)] 

                    elif noise[i][j] < 0.4 * depth: # Sand
                        colour_noise[i][j][k] = [255, 255, int(255 * noise_unormalized)] 

                    elif noise[i][j] < 0.6 * depth: # Grass
                        colour_noise[i][j][k] = [0, int(255 * noise_unormalized), 0]

                    elif noise[i][j] < 0.9 * depth: # Mountains
                        colour_noise[i][j][k] = [200 - int(200 * noise_unormalized), 200 - int(200 * noise_unormalized), 0] 

                    else: # Snow
                        colour_noise[i][j][k] = [int(255 * noise_unormalized), int(255 * noise_unormalized), int(255 * noise_unormalized)]

    colour_noise = colour_noise / 255
    return colour_noise


# Parameters
rows = 50 # For resolution
columns = 50 # For resolution
depth = 50 # For resolution
scale = rows / 4 # For zoom
octaves = 4 # How many levels
persistence = 0.5 # How much less each octave contributes
lacunarity = 2.75 # For detail
factor = 0 # -0.2 for islandy terrain, 0 for normal, 0.2 for rocky terrain


noise = noise_array()
noise = normalize()


# Create 3D voxel grid (False = empty, True = solid voxel)
voxels = np.zeros((rows, columns, depth))

for i in range(rows):
    for j in range(columns):
        voxels[i, j, :int(noise[i][j])] = True  # Fill from bottom up to terrain height

colours = colourmap()


# Plot
ax = plt.figure().add_subplot(projection = '3d')
ax.voxels(voxels, facecolors = colours)
plt.show()
