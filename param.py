import os

# Arguments
args = {
    "INPUT_PATH": '../reddots/',
    "MODEL_PATH": './saved_model',
    "SAVED_SP": 'sp_region{}.pickle',
    "SAVED_TM": 'tm_region{}.pickle'
}

# I/O files
input_dir = os.path.abspath(args["INPUT_PATH"])
sp_model = os.path.join(args["MODEL_PATH"], args["SAVED_SP"])
tm_model = os.path.join(args["MODEL_PATH"], args["SAVED_TM"])

# Model Parameters
default_parameters = {'enc': {'featureCount': 27,
         'resolution': 0.5,
         'size': 100,
         'sparsity': 0.2},
 'epochs': 5,
 'ratio': 0.6,
 'sp': {'boostStrength': 3.875401703739625,
        'columnCount': 1500,
        'localAreaDensity': 0.045686323733035716,
        'potentialPct': 0.9599047766082827,
        'potentialRadius': 1500,
        'synPermActiveInc': 0.03860930947392123,
        'synPermConnected': 0.16130391936570504,
        'synPermInactiveDec': 0.007744449152507922},
 'tm': {'activationThreshold': 16,
        'cellsPerColumn': 7,
        'initialPerm': 0.15539697963997492,
        'maxSegmentsPerCell': 210,
        'maxSynapsesPerSegment': 43,
        'minThreshold': 12,
        'newSynapseCount': 24,
        'permanenceDec': 0.08068640126739315,
        'permanenceInc': 0.1033227963647751}}