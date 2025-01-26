import numpy as np
import matplotlib.pyplot as plt

def displayImage(synthesizedTexture, imageName, path):
    plt.imshow(synthesizedTexture)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(path+imageName)
    plt.show()

def randomPlacement(texture, imageName, patchSize=40):
    # take random square blocks from input texture and 
    # place randomly onto syntehsized texture
    # print("Texture shape: ", texture.shape)

    textureHeight, textureWidth, _ = texture.shape
    synthHeight = textureHeight*5
    synthWidth = textureWidth*5
    synthesizedTexture = np.zeros((synthHeight, synthWidth, texture.shape[2]))
    # print("Synthesized texture shape: ", synthesizedTexture.shape)

    for i in range(0, int(synthHeight/patchSize)):
        for j in range(0, int(synthWidth/patchSize)):
            width = np.random.randint(0, textureWidth - patchSize)
            height = np.random.randint(0, textureHeight - patchSize)
            block = texture[height:height+patchSize, width:width+patchSize, :]
            row = i*patchSize
            col = j*patchSize
            synthesizedTexture[row:row+patchSize, col:col+patchSize, :] = block

    displayImage(synthesizedTexture, imageName, './data/results/synthesis/method1/')

#text
imageName = 'radishes.jpg'
imageName = 'weave.jpg'
texture = plt.imread('./data/textures/' + imageName)

if texture.dtype == 'uint8':
    texture = texture/255.0

if (texture.ndim != 3):
    thirdChannel = texture
    texture = np.stack((thirdChannel,)*3, axis=2)

# randomPlacement(texture, imageName)
# overlapNeighbouringBlocks(texture)
minErrorCut(texture)