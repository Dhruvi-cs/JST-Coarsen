#Make sure you set a correct matlab path
#CFLAGS = -I /usr/include/c++/4.6/


EXAMPLE = example/toy
ALPHAN = 0.5
ALPHAT = 0.5
MATLAB_PATH = /raid/R2014A/bin/matlab


demo:
	python netCondense.py $(EXAMPLE) $(ALPHAN) $(ALPHAT) $(MATLAB_PATH)

clean:
	rm -rf ./example/*_final* ./example/*_superNodes ./example/*_superTimes ./example/*_edgeScores ./example/*_timeScores
