import os


def deepupdate(dict_base, other):
  for k, v in other.items():
    if isinstance(v, dict) and k in dict_base:
      deepupdate(dict_base[k], v)
    else:
      dict_base[k] = v

# Arguments
args = {
    "REGION_NUM": 1,
    "VIZ_COLS": True,
    "DO_LEARNING": True,
    "USE_OLD_MODEL": False,
    "SAVE_MODEL": False,
    "INPUT_PATH": 'data/',
    "MODEL_PATH": '../saved_model',
    "SAVED_SP": 'sp_region{}.pickle',
    "SAVED_TM": 'tm_region{}.pickle'
}

# I/O files
input_file = os.path.abspath(args["INPUT_PATH"])
sp_model = os.path.join(args["MODEL_PATH"], args["SAVED_SP"])
tm_model = os.path.join(args["MODEL_PATH"], args["SAVED_TM"])

# Model Parameters
parameters = {
    'enc': {
        'resolution': 0.88,
        'sparsity': 0.02
    },
    'sdrc_alpha': 0.1,
    'sp': {
        'potentialPct': 1.0,
        'wrapAround': True,
        'localAreaDensity': .02,
        'synPermInactiveDec': 0.01,
        'synPermActiveInc': 0.1,
        'synPermConnected': 0.5,
        'boostStrength': 0.0,
    },
    'tm': {
        'initialPermanence': 0.21,
        'connectedPermanence': 0.5,
        'maxSegmentsPerCell': 128,
        'permanenceDecrement': 0.3,
        'permanenceIncrement': 0.3,
        'predictedSegmentDecrement': 0.0,
        'activationThreshold': 4,
        'cellsPerColumn': 16,
        'maxSynapsesPerSegment': 32,
        'minThreshold': 2,
        'maxNewSynapseCount': 20,
    },
}