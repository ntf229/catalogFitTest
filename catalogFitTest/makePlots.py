"""
Makes plots from numpy array data,
must run storeData.py first to generate numpy arrays
"""

from utils import *

plotFits = False

for m in range(len(methods)): 
    if m == 1:
        massPlotIndex = 0
    elif m == 2:
        SFRPlotIndex = 0
    elif m == 3:
        massPlotIndex = 1
        SFRPlotIndex = 1
    for d in range(len(dustPaths)): 
        if (m == 0) and (d == 0):
            continue # SFH method needs dust turned on
        for b in range(len(dataTypes)): # ['GSWLC1', 'DustPedia']
            if m == 1:
                if d == 3:
                    if b == 0:
                        continue # skip normSFH, power law, GSWLC1 (didn't finish)
            path = dataPath+methods[m]+'/'+dustPaths[d]+'/'+dataTypes[b]+'/'
            bestParams = np.load(path+'bestParams.npy')
            bestParamsUnc = np.load(path+'bestParamsUnc.npy') # len(singleNames), numView, numParams, 2
            bestSFRs = np.load(path+'bestSFRs.npy')
            bestSFRUnc = np.load(path+'bestSFRUnc.npy') # len(singleNames), numView, 2
            paramNames = np.load(path+'paramNames.npy')
            bestFluxes = np.load(path+'bestFluxes.npy')
            bestSpec = np.load(path+'bestSpec.npy')
            obsFluxes = np.load(path+'obsFluxes.npy')
            obsUnc = np.load(path+'obsUnc.npy')
            if b == 0:
                waveEff = wave_eff_GSWLC1
            else:
                waveEff = wave_eff_DustPedia
            if plotMass[m]:
                fitMasses = bestParams[:,:, paramNames=='total_mass'][:,:,0]
                allBestMasses[:, m, d, b, :] = fitMasses
                allBestMassesUnc[:, m, d, b, 0, :] = bestParamsUnc[:,0,paramNames=='total_mass',:][:,0,:]
                allBestMassesUnc[:, m, d, b, 1, :] = bestParamsUnc[:,1,paramNames=='total_mass',:][:,0,:]
                # low mass moments
                lowMassFaceLogResidual = np.log10(fitMasses[lowMass, 0]) - np.log10(singleStellarMass[lowMass])
                lowMassEdgeLogResidual = np.log10(fitMasses[lowMass, 1]) - np.log10(singleStellarMass[lowMass])
                lowMassBiasMass[massPlotIndex, d,b,0] = np.mean(lowMassFaceLogResidual) # face
                lowMassBiasMass[massPlotIndex, d,b,1] = np.mean(lowMassEdgeLogResidual) # edge
                lowMassStdMass[massPlotIndex, d,b,0] = np.sqrt(np.mean(
                    lowMassFaceLogResidual**2) - np.mean(lowMassFaceLogResidual)**2) 
                lowMassStdMass[massPlotIndex, d,b,1] = np.sqrt(np.mean(
                    lowMassEdgeLogResidual**2) - np.mean(lowMassEdgeLogResidual)**2)
                # high mass moments
                highMassFaceLogResidual = np.log10(fitMasses[highMass, 0]) - np.log10(singleStellarMass[highMass])
                highMassEdgeLogResidual = np.log10(fitMasses[highMass, 1]) - np.log10(singleStellarMass[highMass])
                highMassBiasMass[massPlotIndex, d,b,0] = np.mean(highMassFaceLogResidual) # face
                highMassBiasMass[massPlotIndex, d,b,1] = np.mean(highMassEdgeLogResidual) # edge
                highMassStdMass[massPlotIndex, d,b,0] = np.sqrt(np.mean(
                    highMassFaceLogResidual**2) - np.mean(highMassFaceLogResidual)**2) 
                highMassStdMass[massPlotIndex, d,b,1] = np.sqrt(np.mean(
                    highMassEdgeLogResidual**2) - np.mean(highMassEdgeLogResidual)**2)
            if plotSFR[m]:
                allBestSFRs[:, m, d, b, :] = bestSFRs
                allBestSFRsUnc[:, m, d, b, 0, :] = bestSFRUnc[:, 0, :]
                allBestSFRsUnc[:, m, d, b, 1, :] = bestSFRUnc[:, 1, :]
                # low mass moments
                lowMassFaceLogResidual = np.log10(bestSFRs[lowMass, 0]) - np.log10(singleSFR[lowMass])
                lowMassEdgeLogResidual = np.log10(bestSFRs[lowMass, 1]) - np.log10(singleSFR[lowMass])
                lowMassBiasSFR[SFRPlotIndex, d,b,0] = np.mean(lowMassFaceLogResidual) # face
                lowMassBiasSFR[SFRPlotIndex, d,b,1] = np.mean(lowMassEdgeLogResidual) # edge
                lowMassStdSFR[SFRPlotIndex, d,b,0] = np.sqrt(np.mean(
                    lowMassFaceLogResidual**2) - np.mean(lowMassFaceLogResidual)**2) 
                lowMassStdSFR[SFRPlotIndex, d,b,1] = np.sqrt(np.mean(
                    lowMassEdgeLogResidual**2) - np.mean(lowMassEdgeLogResidual)**2)
                # high mass moments
                highMassFaceLogResidual = np.log10(bestSFRs[highMass, 0]) - np.log10(singleSFR[highMass])
                highMassEdgeLogResidual = np.log10(bestSFRs[highMass, 1]) - np.log10(singleSFR[highMass])
                highMassBiasSFR[SFRPlotIndex, d,b,0] = np.mean(highMassFaceLogResidual) # face
                highMassBiasSFR[SFRPlotIndex, d,b,1] = np.mean(highMassEdgeLogResidual) # edge
                highMassStdSFR[SFRPlotIndex, d,b,0] = np.sqrt(np.mean(
                    highMassFaceLogResidual**2) - np.mean(highMassFaceLogResidual)**2) 
                highMassStdSFR[SFRPlotIndex, d,b,1] = np.sqrt(np.mean(
                    highMassEdgeLogResidual**2) - np.mean(highMassEdgeLogResidual)**2)
            # calculate chi square
            for g in range(len(singleNames)):
                if ((m == 0) and (d == 3)) and ((b == 1) and (singleNames[g] == 'g1.64e11')): 
                    continue # this galaxy didn't finish
                if b == 1: # UV-IR only
                    chi2_UV_IR[g,m,d,0,:] = ((obsFluxes[g,0,:] - bestFluxes[g,0,:])**2) / obsUnc[g,0,:]**2  # face-on
                    chi2_UV_IR[g,m,d,1,:] = ((obsFluxes[g,1,:] - bestFluxes[g,1,:])**2) / obsUnc[g,1,:]**2  # edge-on
                    chi2_UV_Optical[g,m,d,0,:,b] = ((obsFluxes[g,0,:len(filterlist_GSWLC1)] - 
                        bestFluxes[g,0,:len(filterlist_GSWLC1)])**2) / obsUnc[g,0,:len(filterlist_GSWLC1)]**2  # face-on
                    chi2_UV_Optical[g,m,d,1,:,b] = ((obsFluxes[g,1,:len(filterlist_GSWLC1)] - 
                        bestFluxes[g,1,:len(filterlist_GSWLC1)])**2) / obsUnc[g,1,:len(filterlist_GSWLC1)]**2  # edge-on
                else:
                    chi2_UV_Optical[g,m,d,0,:,b] = ((obsFluxes[g,0,:] - bestFluxes[g,0,:])**2) / obsUnc[g,0,:]**2  # face-on
                    chi2_UV_Optical[g,m,d,1,:,b] = ((obsFluxes[g,1,:] - bestFluxes[g,1,:])**2) / obsUnc[g,1,:]**2  # edge-on
            if plotFits:
                for g in range(len(singleNames)):
                    if (singleNames[g] != 'g1.52e11') and (singleNames[g] != 'g5.02e11'):
                        continue
                    galaxy = singleNames[g]
                    nameMask = names == galaxy
                    edgeIndex = np.argmin(axisRatios[nameMask])
                    faceIndex = np.argmax(axisRatios[nameMask])
                    if d == 0:
                        shiftedCatalogFaceSpec = (100**2 * catalogSpectrum_nodust[nameMask][faceIndex]) / dist[g]**2
                        shiftedCatalogEdgeSpec = (100**2 * catalogSpectrum_nodust[nameMask][edgeIndex]) / dist[g]**2
                        catalogWave = wave_nodust
                    else:
                        shiftedCatalogFaceSpec = (100**2 * catalogSpectrum[nameMask][faceIndex]) / dist[g]**2
                        shiftedCatalogEdgeSpec = (100**2 * catalogSpectrum[nameMask][edgeIndex]) / dist[g]**2
                        catalogWave = wave
                    plotFit(m, d, waveEff, obsFluxes[g,0,:], catalogWave, shiftedCatalogFaceSpec, 
                            bestFluxes[g,0,:], obsUnc[g,0,:], fspsWave, bestSpec[g,0,:], galaxy, 'face')
                    plotFit(m, d, waveEff, obsFluxes[g,1,:], catalogWave, shiftedCatalogEdgeSpec, 
                            bestFluxes[g,1,:], obsUnc[g,1,:], fspsWave, bestSpec[g,1,:], galaxy, 'edge')

plotAllResiduals('total_mass', singleStellarMass, allBestMasses, allBestMassesUnc)
plotAllResiduals('SFR', singleSFR, allBestSFRs, allBestSFRsUnc)
plotMoments(methods, highMassBiasMass, highMassStdMass, highMassBiasSFR, highMassStdSFR)
plotChi2(chi2_UV_IR, chi2_UV_Optical, wave_eff_DustPedia, wave_eff_GSWLC1)
plotMassSFR(singleStellarMass, singleSFR, allBestMasses, allBestSFRs)

print('done')





