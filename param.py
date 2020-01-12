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
default_parameters = {
    'enc': {'featureCount': 14,
          'resolution': 0.00011317646514578243,
          'size': 102,
          'sparsity': 0.02077283529660421},
    'epochs': 5,
    'ratio': 0.4948194009534702,
    'sp': {'boostStrength': 3.319583347002143,
        'columnCount': 1374,
        'localAreaDensity': 0.04634599109721217,
        'potentialPct': 0.981789258715654,
        'potentialRadius': 1316,
        'synPermActiveInc': 0.04127044187760044,
        'synPermConnected': 0.1696263968164441,
        'synPermInactiveDec': 0.007075129418791205},
    'tm': {'activationThreshold': 17,
        'cellsPerColumn': 8,
        'initialPerm': 0.1755380706835584,
        'maxSegmentsPerCell': 184,
        'maxSynapsesPerSegment': 49,
        'minThreshold': 12,
        'newSynapseCount': 25,
        'permanenceDec': 0.08500326946369395,
        'permanenceInc': 0.11137303572300684}
}