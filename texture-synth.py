import numpy as np
import matplotlib.pyplot as plt

def randomPlacement(texture, patchSize=40, dim=400):
    # take random square blocks from input texture and 
    # place randomly onto syntehsized texture
    print("Texture shape: ", texture.shape)

    textureHeight, textureWidth, _ = texture.shape
    synthesizedTexture = np.zeros((dim, dim, texture.shape[2]))
    print("Synthesized texture shape: ", synthesizedTexture.shape)

    for i in range(0, int(dim/patchSize)):
        for j in range(0, int(dim/patchSize)):
            width = np.random.randint(0, textureWidth - patchSize)
            height = np.random.randint(0, textureHeight - patchSize)
            block = texture[height:height+patchSize, width:width+patchSize, :]
            row = i*patchSize
            col = j*patchSize
            synthesizedTexture[row:row+patchSize, col:col+patchSize, :] = block
    return synthesizedTexture


#text
imageName = 'radishes.jpg'
texture = plt.imread('./data/textures/' + imageName)
print("texture d type: ", texture.dtype)
if texture.dtype == 'uint8':
    texture = texture/255.0

synthesizedTexture = randomPlacement(texture)

plt.imshow(synthesizedTexture)
plt.axis('off')
plt.show()