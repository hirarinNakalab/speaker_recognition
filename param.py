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
default_parameters = {'enc': {'featureCount': 30,
         'resolution': 0.35764051992372875,
         'size': 74,
         'sparsity': 0.22212565698924117},
 'epochs': 7,
 'ratio': 0.7343840172724385,
 'sp': {'boostStrength': 3.5594018728423693,
        'columnCount': 1604,
        'localAreaDensity': 0.046174898827391855,
        'potentialPct': 0.9178993643504982,
        'potentialRadius': 1888,
        'synPermActiveInc': 0.038518953699589575,
        'synPermConnected': 0.1560844133229486,
        'synPermInactiveDec': 0.009801738140268376},
 'tm': {'activationThreshold': 12,
        'cellsPerColumn': 7,
        'initialPerm': 0.16456677162557537,
        'maxSegmentsPerCell': 169,
        'maxSynapsesPerSegment': 51,
        'minThreshold': 12,
        'newSynapseCount': 18,
        'permanenceDec': 0.08129193530197774,
        'permanenceInc': 0.07575405585662963}}