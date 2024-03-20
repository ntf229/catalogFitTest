This project uses the NIHAO-SKIRT-Catalog to test the accuracy of mass and SFR inferences
from the SED modeling package Prospector. 

This repository contains numpy arrays with the best fitting (maximum posterior density) 
samples from Prospector fits using the DYNESTY dynamic nested sampling package. 

To reproduce our analysis, first go to utils.py and change the home variable to
this repo's location, then run the makePlots.py file.
