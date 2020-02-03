import os

# Arguments
args = {
    "INPUT_PATH": '../reddots/',
    "MODEL_PATH": './saved_model',
    "SAVED_SP": 'sp.pickle',
    "SAVED_TM": 'tm.pickle'
}

# I/O files
input_dir = os.path.abspath(args["INPUT_PATH"])
sp_model = os.path.join(args["MODEL_PATH"], args["SAVED_SP"])
tm_model = os.path.join(args["MODEL_PATH"], args["SAVED_TM"])

# Model Parameters
default_parameters = {'enc': {'featureCount': 25,
         'resolution': 0.36695746068801893,
         'size': 68,
         'sparsity': 0.21955891357499074},
 'epochs': 9,
 'ratio': 0.6167930961024468,
 'sp': {'boostStrength': 3.1192185561800443,
        'columnCount': 1461,
        'localAreaDensity': 0.04746850349383391,
        'potentialPct': 0.8157461351372742,
        'potentialRadius': 2239,
        'synPermActiveInc': 0.03453430923170417,
        'synPermConnected': 0.14792087948214822,
        'synPermInactiveDec': 0.010501527343828226},
 'tm': {'activationThreshold': 11,
        'cellsPerColumn': 8,
        'initialPerm': 0.20840260761920143,
        'maxSegmentsPerCell': 149,
        'maxSynapsesPerSegment': 54,
        'minThreshold': 12,
        'newSynapseCount': 20,
        'permanenceDec': 0.07044642988973361,
        'permanenceInc': 0.08306790785000162}}