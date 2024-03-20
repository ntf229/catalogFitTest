"""
Saves select Prospector fit data as numpy arrays
"""

from utils import *

for m in range(len(methods)): 
    for d in range(len(dustPaths)): 
        if (m == 0) and (d == 0):
            continue # SFH method needs dust turned on
        for b in range(len(dataTypes)): # ['GSWLC1', 'DustPedia']
            if m == 1:
                if d == 3:
                    if b == 0:
                        continue # skip normSFH, power law, GSWLC1 (didn't finish)
            path = dataPath+methods[m]+'/'+dustPaths[d]+'/'+dataTypes[b]+'/'
            os.system('mkdir -p '+path)
            numParams = int(numMethod[m] + numDust[d])
            bestParams = np.zeros((len(singleNames), numView, numParams)) 
            bestParamsUnc = np.zeros((len(singleNames), numView, numParams, 2)) # [lower, upper]
            bestSFRs = np.zeros((len(singleNames), numView))
            bestSFRUnc = np.zeros((len(singleNames), numView, 2)) # [lower, upper]
            bestFluxes = np.zeros((len(singleNames), numView, numFlux[b]))
            bestSpec = np.zeros((len(singleNames), numView, len(fspsWave)))
            obsFluxes = np.zeros((len(singleNames), numView, numFlux[b]))
            obsUnc = np.zeros((len(singleNames), numView, numFlux[b]))
            for g in range(len(singleNames)):
                if ((m == 0) and (d == 3)) and ((b == 1) and (singleNames[g] == 'g1.64e11')): 
                    continue # this galaxy didn't finish
                (faceParams, faceParamsUnc, faceSFR, faceSFRUnc, 
                 edgeParams, edgeParamsUnc, edgeSFR, edgeSFRUnc, 
                 paramNames, faceFluxes, edgeFluxes, faceObs, faceUnc, 
                 edgeObs, edgeUnc, faceSpec, edgeSpec) = getFitData(
                    fitPath+methods[m]+'/dynesty/'+dustPaths[d]+dataTypes[b]+'/'+singleNames[g]+'/')
                bestParams[g,0,:] = faceParams  
                bestParams[g,1,:] = edgeParams 
                bestParamsUnc[g,0,:,:] = faceParamsUnc  
                bestParamsUnc[g,1,:,:] = edgeParamsUnc 
                bestSFRs[g,0] = faceSFR
                bestSFRs[g,1] = edgeSFR
                bestSFRUnc[g,0,:] = faceSFRUnc
                bestSFRUnc[g,1,:] = edgeSFRUnc
                bestFluxes[g,0,:] = faceFluxes
                bestFluxes[g,1,:] = edgeFluxes
                bestSpec[g,0,:] = faceSpec 
                bestSpec[g,1,:] = edgeSpec 
                obsFluxes[g,0,:] = faceObs
                obsFluxes[g,1,:] = edgeObs
                obsUnc[g,0,:] = faceUnc
                obsUnc[g,1,:] = edgeUnc
            np.save(path+'bestParams.npy', bestParams)
            np.save(path+'bestParamsUnc.npy', bestParamsUnc)
            np.save(path+'bestSFRs.npy', bestSFRs)
            np.save(path+'bestSFRUnc.npy', bestSFRUnc)
            np.save(path+'paramNames.npy', paramNames)
            np.save(path+'bestFluxes.npy', bestFluxes)
            np.save(path+'bestSpec.npy', bestSpec)
            np.save(path+'obsFluxes.npy', obsFluxes)
            np.save(path+'obsUnc.npy', obsUnc)


