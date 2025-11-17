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

    return (noise - mini + factor) / (maxi - mini)


def colourmap(): 
    colour_noise = np.zeros((rows, columns, 3), dtype=np.uint8)
                        
    for i in range(rows):
        for j in range(columns):
            if earth and math.sqrt(math.pow((i - 250), 2) + math.pow((j - 250), 2)) > 200: # Sky
                colour_noise[i][j] = [1, 1, 1]

            elif noise[i][j] < 0.45: # Sea
                colour_noise[i][j] = [0, 0, int(255 * noise[i][j])] 

            elif noise[i][j] < 0.5: # Sand
                colour_noise[i][j] = [255, 255, int(255 * noise[i][j])] 

            elif noise[i][j] < 0.65: # Grass
                colour_noise[i][j] = [0, int(255 * noise[i][j]), 0]

            elif noise[i][j] < 0.85: # Mountains
                colour_noise[i][j] = [200 - int(200 * noise[i][j]), 200 - int(200 * noise[i][j]), 0] 

            else: # Snow
                colour_noise[i][j] = [int(255 * noise[i][j]), int(255 * noise[i][j]), int(255 * noise[i][j])]

    return colour_noise


# Parameters
rows = 500 # For resolution
columns = 500 # For resolution
scale = rows / 4 # For zoom
octaves = 4 # How many levels
persistence = 0.5 # How much less each octave contributes
lacunarity = 2.75 # For detail
factor = 0 # -0.2 for islandy terrain, 0 for normal, 0.2 for rocky terrain
earth = 0 # 0 For no earth effect, 1 for earth effect


noise = noise_array()
noise = normalize()
noise = colourmap()


# Plot
plt.imshow(noise)
plt.axis('off')
plt.show()
