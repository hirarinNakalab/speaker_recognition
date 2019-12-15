import os

dimension = 2

# Arguments
args = {
    "REGION_NUM": 1,
    "VIZ_COLS": False,
    "DO_LEARNING": True,
    "USE_OLD_MODEL": False,
    "SAVE_MODEL": False,
    "INPUT_PATH": 'data/',
    "OUTPUT_PATH": 'outputs/',
    "MODEL_PATH": '../saved_model',
    "IMG_PATH": './',
    "CSV_FILE": '*.csv',
    "OUTPUT_FILE": '{}_output.csv',
    "OUTPUT_IMG": 'step{}.png',
    "SAVED_SP": 'sp_region{}.pickle',
    "SAVED_TM": 'tm_region{}.pickle'
}
if dimension == 1:
    additional_dict = {
        "INPUT_DIM": 60,
        "COLUMN_COUNT": 1638,
        "CELLS_COUNT": 13,
    }
elif dimension == 2:
    additional_dict = {
        "SDR_DIM": 50,
        "EMB_DIM": 50,
        "LAYER_DIM": (40, 40),
    }
args.update(additional_dict)

# I/O files
input_file = os.path.abspath(args["INPUT_PATH"])
output_file = os.path.join(args["OUTPUT_PATH"], args["OUTPUT_FILE"])
output_img = os.path.join(args["IMG_PATH"], args["OUTPUT_IMG"])
result_csv = os.path.abspath(os.path.join(args["OUTPUT_PATH"], args["CSV_FILE"]))
sp_model = os.path.join(args["MODEL_PATH"], args["SAVED_SP"])
tm_model = os.path.join(args["MODEL_PATH"], args["SAVED_TM"])


# Model Parameters
input_shape = (args["SDR_DIM"], args["EMB_DIM"]) if dimension == 2 else (args["INPUT_DIM"],)
layer_shape = args["LAYER_DIM"] if dimension == 2 else (args["COLUMN_COUNT"],)

parameters = {
    'enc': {
        'resolution': 0.88,
    },
    'sdrc_alpha': 0.1,
    'sp': {
        'inputDimensions': input_shape,
        'columnDimensions': layer_shape,
        'potentialRadius': int(input_shape[0] * 3 / 8),
        'wrapAround': True,
    },
    'tm': {
        'initialPermanence': 0.21,
        'connectedPermanence': 0.5,
        'maxSegmentsPerCell': 128,
        'permanenceDecrement': 0.1,
        'permanenceIncrement': 0.1,
        'predictedSegmentDecrement': 0.0,
    },
}

if dimension == 1:
    upd_parameters = {
        'enc': {
            'size': args["INPUT_DIM"],
            'sparsity': 0.05
        },
        'sp': {
            'potentialPct': 0.85,
            'globalInhibition': True,
            'localAreaDensity': 0.04395604395604396,
            'synPermInactiveDec': 0.006,
            'synPermActiveInc': 0.04,
            'synPermConnected': 0.13999999999999999,
            'boostStrength': 3.0,
        },
        'tm': {
            'columnDimensions': (args["COLUMN_COUNT"],),
            'activationThreshold': 17,
            'cellsPerColumn': args["CELLS_COUNT"],
            'maxSynapsesPerSegment': 64,
            'minThreshold': 10,
            'maxNewSynapseCount': 32,
        }
    }
else:
    upd_parameters = {
        'enc': {
            'size': args["SDR_DIM"],
            'sparsity': 0.02
        },
        'sp': {
            'potentialPct': 1.0,
            'globalInhibition': False,
            'localAreaDensity': .02,
            'synPermInactiveDec': 0.01,
            'synPermActiveInc': 0.1,
            'synPermConnected': 0.5,
            'boostStrength': 0.0,
        },
        'tm': {
            'columnDimensions': layer_shape,
            'activationThreshold': 16,
            'cellsPerColumn': 16,
            'maxSynapsesPerSegment': 32,
            'minThreshold': 12,
            'maxNewSynapseCount': 20,
        }
    }

def deepupdate(dict_base, other):
  for k, v in other.items():
    if isinstance(v, dict) and k in dict_base:
      deepupdate(dict_base[k], v)
    else:
      dict_base[k] = v

deepupdate(parameters, upd_parameters)