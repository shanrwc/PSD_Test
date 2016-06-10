#! /usr/bin/env python

import os
import sys
import fileinput
import string

#############################################################################   
##########################Start Main Body of Code############################   
#############################################################################   

#set up list of plots to insert                                                 
plots = []

#read in lines from template                                                    
template = "template.tex"
input = open(template,"r")
lines = input.readlines()

#make outfile, which will be compiled by LaTex.                                 
cutflow = "slides.tex"
if os.path.isfile(cutflow): os.remove(cutflow)
cutflow = open(cutflow,"w")

#loop over template lines.  If line contains "cutflow.txt," open that cutflow file                                      
#and put cutflow lines there.                                                                                           
#Otherwise, just print out template line.                                                                               
for line in lines:
    if ".txt" in line or ".tex" in line:
        if "%" in line: continue
        textFile = open("tables/"+ line.strip(),"r")
        moreInput = textFile.readlines()
        for info in moreInput:
            if "onej" in line and "1j" not in info: continue
            if "h150" in line: info = info.replace("h150","higgs")
            cutflow.write(info)
    else:
        cutflow.write(line)

cutflow.close()
