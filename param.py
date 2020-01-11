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
    'ratio': 0.4,
    'epochs': 5,
    'enc': {'featureCount': 16,
         'resolution': 0.000149999993,
         'size': 85,
         'sparsity': 0.017065233011050153},
    'sp': {'boostStrength': 2.672350211791666,
        'columnCount': 1722,
        'potentialRadius': 1722,
        'localAreaDensity': 0.04641347006804628,
        'potentialPct': 0.9824769324077725,
        'synPermActiveInc': 0.04130319201519869,
        'synPermConnected': 0.13894113423516888,
        'synPermInactiveDec': 0.005703946076797715},
    'tm': {'activationThreshold': 17,
        'cellsPerColumn': 11,
        'initialPerm': 0.2011464529507132,
        'maxSegmentsPerCell': 145,
        'maxSynapsesPerSegment': 57,
        'minThreshold': 12,
        'newSynapseCount': 25,
        'permanenceDec': 0.07895233493179421,
        'permanenceInc': 0.11398959870861339}
}