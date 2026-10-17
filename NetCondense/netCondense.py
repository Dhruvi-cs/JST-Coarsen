# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 11:21:25 2016

@author: Bijaya Adhikari
"""

import sys
import time
import subprocess
import os

from merge import netCondenseRoutine



if len(sys.argv)!=5:
    print('Check usage')
    
pathToFile = sys.argv[1]
alphaN = sys.argv[2]
alphaT = sys.argv[3]
matlab_path=sys.argv[4]
t1=time.time()


scoringFunction = "getScores('" + pathToFile + "')"

# Determine whether the executable is Octave or MATLAB and set flags accordingly.
exe_name = os.path.basename(matlab_path).lower()
print(f"Detected executable: {exe_name}")
if 'octave' in exe_name:
    # Octave prefers long GNU-style flags and uses --eval to run code
    cmd = [matlab_path, '--no-gui', '--no-window-system', '--eval', scoringFunction]
    print("Detected Octave executable. Using Octave flags.")
else:
    # Assume MATLAB-style invocation
    cmd = [matlab_path, '-nosplash', '-nojvm', '-r', scoringFunction]
    print("Assuming MATLAB executable. Using MATLAB flags.")

print(f"Running external command: {' '.join(cmd)}")
try:
    res = subprocess.run(cmd, check=True, capture_output=True, text=True)
    # print output for debugging if needed
    if res.stdout:
        print(res.stdout)
    if res.stderr:
        print(res.stderr)
except subprocess.CalledProcessError as e:
    print(f"External process failed with return code {e.returncode}")
    if hasattr(e, 'stdout') and e.stdout:
        print('STDOUT:\n', e.stdout)
    if hasattr(e, 'stderr') and e.stderr:
        print('STDERR:\n', e.stderr)
    raise

netCondenseRoutine(pathToFile, float(alphaN), float(alphaT), pathToFile+'_edgeScores', pathToFile+'_timeScores')

t2 = time.time()

print(t2-t1)



    

