# Startup Press Release Analyzer  

##TL;DR  
blah blah blah


##Project Overview  
**Project Goals**  
The goal was to do an analysis of startup press releases to see if there were any common trends among them.

**The Process**  
I began with a list of all the companies on Crunchbase. This served to represent as a substantive representation of most startup companies.

**Results**  
results here

##Detailed Process
**Algorithms/Techniques**  
tf-idf
NMF

**Validation**  
Custom n_components grid search
Insert plot here of MSE vs n_components

**Insights**  
Explain top topics and how they map to typical startup industries.

##Detailed Code Walk Through
**data_acquisition.py**  
walkthru here

**data_munging.py**  
walkthru here

**model.py**  
walkthru here

**visualizations.ipynb**  
walkthru here

##How to run the program
If you had access to the Crunchbase list of companies csv file and the MongoDB database containing the press releases, you would simply call 'run model.py', and the entire program would run and return to you the top 15 words from the top eight latent topics.