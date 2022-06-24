CC = g++
CFLAGS = -g -std=c++17 -march=native -ggdb -O2 -fPIC -fopenmp -Wall
PY = python3

SETTINGS = settings
GEN = gen
PF = pf
PLT = plt
EM = em
EMNG = emng

INCLUDES = -L pips/lib \
			-l c++-pips-core -l amrl_shared_lib -l z3 \
			-I pips/src -I pf_custom -I accSim -I pips/submodules/json/single_include/


.SILENT: $(SETTINGS) $(GEN) $(PF) $(PLT) $(EMNG) $(EM) clean

$(SETTINGS):
			$(CC) $(CFLAGS) -o ts translateSettings.cpp
			./ts settings

$(GEN):
			$(MAKE) $(SETTINGS) && \
			mkdir -p accSim/out && \
			$(CC) $(CFLAGS) -o accSim/out/gen accSim/generate.cpp $(INCLUDES) && \
			accSim/out/gen && \
			cp accSim/out/data.json pips/examples/data.json
$(PF):
			$(MAKE) $(SETTINGS) && \
			mkdir -p particleFilter/out && \
			$(CC) $(CFLAGS) -o particleFilter/out/pf particleFilter/pf_runner.cpp && \
			particleFilter/out/pf

$(PLT):
			$(MAKE) $(SETTINGS) && \
			mkdir -p synthesis/plots && \
			$(PY) particleFilter/plotter.py

$(EMNG):
			rm -rf synthesis/out && \
			mkdir -p synthesis/out/examples && \
			$(CC) $(CFLAGS) synthesis/em.cpp $(INCLUDES) -o synthesis/out/em && \
			synthesis/out/em

$(EM):
			$(MAKE) $(GEN) && \
			$(MAKE) $(EMNG)

clean: 
			rm -rf accSim/out \
					particleFilter/out \
					particleFilter/plots \
					synthesis/plots \
					synthesis/out
