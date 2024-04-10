This project uses the [NIHAO-SKIRT-Catalog](https://github.com/ntf229/NIHAO-SKIRT-Catalog "NIHAO-SKIRT-Catalog")  
to test the accuracy of mass and star-formation rate (SFR) inferences
from the SED modeling package [Prospector](https://github.com/bd-j/prospector "Prospector").   

This repository contains numpy arrays with the best fitting (maximum posterior density) 
samples from Prospector fits using the DYNESTY dynamic nested sampling package. 

To reproduce the analysis of  
"Testing the accuracy of SED modeling techniques using the NIHAO-SKIRT-Catalog" (in prep.),    
first go to utils.py and change the home variable to  
this repositories's location, then run the makePlots.py file.
