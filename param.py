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
default_parameters = {
 'enc': {
     'resolution': 0.88,
     'size': 100,
     'sparsity': 0.02,
     'featureCount': 15,
 },
 'sp': {'boostStrength': 3.0,
        'columnCount': 1638,
        'localAreaDensity': 0.04395604395604396,
        'potentialPct': 0.85,
        'synPermActiveInc': 0.04,
        'synPermConnected': 0.13999999999999999,
        'synPermInactiveDec': 0.006,
        'wrapAround': True},
 'tm': {'activationThreshold': 17,
        'cellsPerColumn': 13,
        'initialPerm': 0.21,
        'maxSegmentsPerCell': 128,
        'maxSynapsesPerSegment': 64,
        'minThreshold': 10,
        'newSynapseCount': 32,
        'permanenceDec': 0.1,
        'permanenceInc': 0.1},
}