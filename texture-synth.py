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

def overlapNeighbouringBlocks(texture, patchSize=80):
    textureHeight, textureWidth, _ = texture.shape
    synthHeight = textureHeight*5
    synthWidth = textureWidth*5
    synthesizedTexture = np.zeros((synthHeight, synthWidth, 4)) +255
    synthesizedTexture [:, :, 3] = 1
    stride = 1
    sbDimWidth = int((textureWidth - patchSize + 1) /stride)
    sbDimHeight = int((textureHeight - patchSize + 1) /stride)
    Sb = np.zeros((sbDimWidth*patchSize, sbDimHeight*patchSize, 4))

    print("Sb shape: ", Sb.shape)
    print("texture shape: ", texture.shape)
    print("synthesized texture shape: ", synthesizedTexture.shape)

    # create all possible patches
    for i in range (0,  sbDimWidth * stride, stride): 
        for j in range(0, sbDimHeight * stride, stride):
            if i + patchSize <= textureHeight and j + patchSize <= textureWidth:
                Sb[i // stride * patchSize:(i // stride + 1) * patchSize, j //stride * patchSize:(j // stride + 1) * patchSize, :3] = texture[i:i + patchSize, j:j + patchSize, :3]

    #place the first patach by picking randomly from the input texture
    initialBlockWidth = np.random.randint(0, textureWidth - patchSize)
    initialBlockHeight = np.random.randint(0, textureHeight - patchSize)
    initialBlock = texture[initialBlockWidth:initialBlockWidth+patchSize,
                            initialBlockHeight:initialBlockHeight+patchSize, :3]
    print("initialBlock shape: ", initialBlock.shape)
    synthesizedTexture[0:patchSize, 0:patchSize, :3] = initialBlock

    #go through and grab patch with smallest overlap error
    #[y, x] = find(errors_of_all_patches <= minimum_error * (tolerance))
    overlapDim = 10
    minErrorBlock = np.zeros((patchSize, patchSize, 3))
    
    for h in range(0, int(synthHeight/patchSize)+overlapDim):
        for w in range(0, int(synthWidth/patchSize)+overlapDim):
            print("h: ", h, " w: ", w)
            
            minError = np.inf
            row = (h*patchSize) - (h*overlapDim)
            col = (w * patchSize) - (w*overlapDim)

            if col+patchSize >= synthesizedTexture.shape[1] or row+patchSize >= synthesizedTexture.shape[0]:
                break

            #overlap region of synthesized texture block
            if h == 0 and w == 0:
                overlapRegionSynthRight = synthesizedTexture [h*patchSize : h*patchSize+patchSize, 
                                                            w*patchSize+patchSize-overlapDim : w*patchSize+patchSize, :3]

            elif h == 0 :
                overlapRegionSynthRight = synthesizedTexture [h*patchSize: h*patchSize+patchSize , 
                                                            (w-1)*(patchSize - overlapDim)+patchSize-overlapDim : (w-1)*(patchSize -overlapDim)+patchSize, :3]
                
            elif w == 0 :
                overlapRegionSynthBot = synthesizedTexture[(h-1)*(patchSize -overlapDim)+patchSize-overlapDim : (h-1)*(patchSize -overlapDim)+patchSize, 
                                                       w*patchSize : w*patchSize+patchSize, :3]
                
            else:
                overlapRegionSynthBot = synthesizedTexture[(h-1)*(patchSize -overlapDim )+patchSize-overlapDim : (h-1)*(patchSize -overlapDim )+patchSize, 
                                                       (w)*(patchSize -overlapDim ) : (w)*(patchSize -overlapDim )+patchSize, :3]
                
                overlapRegionSynthRight = synthesizedTexture [(h)*(patchSize  -overlapDim) :(h)*(patchSize  -overlapDim)+patchSize, 
                                                            (w-1)*(patchSize  -overlapDim)+patchSize-overlapDim : (w-1)*(patchSize  -overlapDim)+patchSize, :3]

            for i in range(sbDimWidth):
                for j in range(sbDimHeight):
                    block = Sb[i*patchSize:i*patchSize+patchSize, j*patchSize:j*patchSize+patchSize, :3]

                    if (h == 0 and w == 0):
                        minErrorBlock = initialBlock
                        continue

                    #top sb edge overlapping
                    overlapRegionBlockTop = block[:overlapDim, :, :3]
                    overlapRegionBlockLeft = block[:, :overlapDim, :3]
                    
                    if h == 0:
                        error = np.sum((overlapRegionSynthRight - overlapRegionBlockLeft)**2)
                    elif w == 0:
                        error = np.sum((overlapRegionSynthBot - overlapRegionBlockTop)**2)
                    else :
                        #2 overlapping regions
                        error = np.sum((overlapRegionSynthRight - overlapRegionBlockLeft)**2) + np.sum((overlapRegionSynthBot - overlapRegionBlockTop)**2)

                    # print("error: ", error)
                    if error < minError :
                        minError = error
                        minErrorBlock = block

            synthesizedTexture[row:row+patchSize, col:col+patchSize, :3] = minErrorBlock 
         
    synthesizedTexture[:, :, 3:4] = synthesizedTexture[:, :, 3:4] * 0.95
    displayImage(synthesizedTexture, imageName, './data/results/synthesis/method2/')


def minErrorCut(texture, patchSize=80):
    textureHeight, textureWidth, _ = texture.shape
    synthHeight = textureHeight*5
    synthWidth = textureWidth*5
    synthesizedTexture = np.zeros((synthHeight, synthWidth, 4)) +255
    synthesizedTexture [:, :, 3] = 1
    stride = 1
    sbDimWidth = int((textureWidth - patchSize + 1) /stride)
    sbDimHeight = int((textureHeight - patchSize + 1) /stride)
    Sb = np.zeros((sbDimWidth*patchSize, sbDimHeight*patchSize, 4))

    print("Sb shape: ", Sb.shape)
    print("texture shape: ", texture.shape)
    print("synthesized texture shape: ", synthesizedTexture.shape)

    # create all possible patches
    for i in range (0,  sbDimWidth * stride, stride): 
        for j in range(0, sbDimHeight * stride, stride):
            if i + patchSize <= textureHeight and j + patchSize <= textureWidth:
                Sb[i // stride * patchSize:(i // stride + 1) * patchSize, j //stride * patchSize:(j // stride + 1) * patchSize, :3] = texture[i:i + patchSize, j:j + patchSize, :3]

    #place the first patach by picking randomly from the input texture
    initialBlockWidth = np.random.randint(0, textureWidth - patchSize)
    initialBlockHeight = np.random.randint(0, textureHeight - patchSize)
    print("initialBlockWidth: ", initialBlockWidth, initialBlockWidth + patchSize)
    print("initialBlockHeight: ", initialBlockHeight, initialBlockHeight + patchSize)

    #!!!!!!!!!
    #hiehgt x height because of random error
    initialBlock = texture[initialBlockHeight:initialBlockHeight+patchSize,
                        initialBlockHeight:initialBlockHeight+patchSize, :3]
    print("initialBlock shape: ", initialBlock.shape)
    synthesizedTexture[0:patchSize, 0:patchSize, :3] = initialBlock

    print("--------------------")

    #go through and grab patch with smallest overlap error
    #[y, x] = find(errors_of_all_patches <= minimum_error * (tolerance))
    overlapDim = 10
    minErrorBlock = np.zeros((patchSize, patchSize, 3))
    tolerance = 0.1
    
    for h in range(0, int(synthHeight/patchSize)+overlapDim):
        for w in range(0, int(synthWidth/patchSize)+overlapDim):
            print("h: ", h, " w: ", w)

            energyMatrix = np.zeros((patchSize, overlapDim))
            M = np.zeros((patchSize, overlapDim))

            # print("energy matrix shape: ", energyMatrix.shape)
            minError = np.inf
            row = (h*patchSize) - (h*overlapDim) 
            col = (w * patchSize) - (w*overlapDim) 

            if col+patchSize >= synthesizedTexture.shape[1] or row+patchSize >= synthesizedTexture.shape[0]:
                break

            #overlap region of synthesized texture block
            if h == 0 and w == 0:
                overlapRegionSynthRight = synthesizedTexture [h*patchSize : h*patchSize+patchSize, 
                                                            w*patchSize+patchSize-overlapDim : w*patchSize+patchSize, :3]

            elif h == 0 :
                overlapRegionSynthRight = synthesizedTexture [h*patchSize: h*patchSize+patchSize , 
                                                            (w-1)*(patchSize - overlapDim)+patchSize-overlapDim : (w-1)*(patchSize -overlapDim)+patchSize, :3]
                
            elif w == 0 :
                overlapRegionSynthBot = synthesizedTexture[(h-1)*(patchSize -overlapDim)+patchSize-overlapDim : (h-1)*(patchSize -overlapDim)+patchSize, 
                                                       w*patchSize : w*patchSize+patchSize, :3]
                
            else:
                overlapRegionSynthBot = synthesizedTexture[(h-1)*(patchSize -overlapDim )+patchSize-overlapDim : (h-1)*(patchSize -overlapDim )+patchSize, 
                                                       (w)*(patchSize -overlapDim ) : (w)*(patchSize -overlapDim )+patchSize, :3]
                
                overlapRegionSynthRight = synthesizedTexture [(h)*(patchSize  -overlapDim) :(h)*(patchSize  -overlapDim)+patchSize, 
                                                            (w-1)*(patchSize  -overlapDim)+patchSize-overlapDim : (w-1)*(patchSize  -overlapDim)+patchSize, :3]

            for i in range(sbDimWidth):
                for j in range(sbDimHeight):
                    block = Sb[i*patchSize:i*patchSize+patchSize, j*patchSize:j*patchSize+patchSize, :3]

                    if (h == 0 and w == 0):
                        minErrorBlock = initialBlock
                        continue

                    #top sb edge overlapping
                    overlapRegionBlockTop = block[:overlapDim, :, :3]
                    overlapRegionBlockLeft = block[:, :overlapDim, :3]
                    
                    if h == 0:
                        error = np.sum((overlapRegionSynthRight - overlapRegionBlockLeft)**2)
                    elif w == 0:
                        error = np.sum((overlapRegionSynthBot - overlapRegionBlockTop)**2)
                    else :
                        #2 overlapping regions
                        error = np.sum((overlapRegionSynthRight - overlapRegionBlockLeft)**2) + np.sum((overlapRegionSynthBot - overlapRegionBlockTop)**2)
                    
                    if error < minError :
                        minError = error
                        minErrorBlock = block
            
            #populate energy matrix
            for i in range(0,patchSize):
                for j in range(0,overlapDim):
                    energyMatrix[i,j] = np.sum((minErrorBlock[i, j, :3] - synthesizedTexture[row+i, col+j, :3])**2)

            #cummilative error matrix
            for i in range(0,patchSize):
                for j in range(0,overlapDim):
                    # print("i: ", i, " j: ", j)
                    if i == 0:
                        M[i,j] = energyMatrix[i,j]
                    else:
                        if j == 0:
                            M[i,j] = energyMatrix[i,j] + min(M[i-1,j], M[i-1,j+1])
                        elif j == overlapDim - 1:
                            M[i,j] = energyMatrix[i,j] + min(M[i-1,j-1], M[i-1,j])
                        else:
                            M[i,j] = energyMatrix[i,j] + min(M[i-1,j-1], M[i-1,j], M[i-1,j+1])
                    
            jMin = int(np.argmin(M[patchSize-1, :]))
            pathVert = [jMin]

            #trace back
            for i in range(patchSize-1, 0, -1):
                j = int(pathVert[-1])

                if j == overlapDim - 1:
                    nextJ = np.argmin([M[i-1, j-1], M[i-1, j]]) +j -1

                elif j == 0:
                    nextJ = np.argmin([M[i-1, j], M[i-1, j+1]])
                else:
                    nextJ =np.argmin([M[i-1, j-1], M[i-1, j], M[i-1, j+1]]) +j -1

                pathVert.append(int(nextJ))
            #stitch the block along the boundary
            pathVert.reverse()

            # print(pathVert)
            # print("=====================================")

            #populate energy matrix
            for i in range(0,patchSize):
                for j in range(0,overlapDim):
                    energyMatrix[i,j] = np.sum((minErrorBlock[i, j, :3] - synthesizedTexture[row+i, col+j, :3])**2)

            # print("energy matrix: ", energyMatrix)

            #cummilative error matrix
            for i in range(0,patchSize):
                for j in range(0,overlapDim):
                    # print("i: ", i, " j: ", j)
                    if i == 0:
                        M[i,j] = energyMatrix[i,j]
                    else:
                        if j == 0:
                            M[i,j] = energyMatrix[i,j] + min(M[i-1,j], M[i-1,j+1])
                        elif j == overlapDim - 1:
                            M[i,j] = energyMatrix[i,j] + min(M[i-1,j-1], M[i-1,j])
                        else:
                            M[i,j] = energyMatrix[i,j] + min(M[i-1,j-1], M[i-1,j], M[i-1,j+1])
                    
            jMin = int(np.argmin(M[patchSize-1, :]))
            pathHorizontal = [jMin]

            #trace back
            for i in range(patchSize-1, 0, -1):
                j = int(pathHorizontal[-1])

                # if j == 0:  # Left edge
                #     nextJ = np.argmin([M[i - 1, j], M[i - 1, j + 1]]) + j
                # elif j == overlapDim - 1:  # Right edge
                #     nextJ = np.argmin([M[i - 1, j - 1], M[i - 1, j]]) + j - 1
                # else:  # Middle
                #     nextJ = np.argmin([M[i - 1, j - 1], M[i - 1, j], M[i - 1, j + 1]]) + j - 1

                if j == overlapDim - 1:
                     nextJ = np.argmin([M[i-1, j-1], M[i-1, j]]) +j -1

                elif j == 0:
                    nextJ = np.argmin([M[i-1, j], M[i-1, j+1]])
                    # nextJ = nextJ + j
                else:
                    nextJ =np.argmin([M[i-1, j-1], M[i-1, j], M[i-1, j+1]]) +j -1

                # print("nextJ: ", nextJ)
                pathHorizontal.append(int(nextJ))

            #stitch the block along the boundary
            pathHorizontal.reverse()

            #attach the block to the synthesized texture with boundary cut
            for i in range(0, patchSize):
                for j in range(0, patchSize):
                    if h == 0 or w == 0:
                        synthesizedTexture[row+i, col+j, :3] = minErrorBlock[i, j, :3]

                    elif j < overlapDim and j <= pathVert[i]:
                        # Leave pixel unchanged (vertical seam)
                        continue
                    elif i < overlapDim and i <= pathHorizontal[j]:
                        # Leave pixel unchanged (horizontal seam)
                        continue
                    else:
                        synthesizedTexture[row+i, col+j, :3] = minErrorBlock[i, j, :3]
                    
            print("=====================================")
         
    displayImage(synthesizedTexture, imageName, './data/results/synthesis/method3/')

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