import os
import pprint
from collections import defaultdict

# I/O files
input_dir = os.path.abspath('../reddots/')
sp_model = '{}_sp.pickle'
tm_model = '{}_tm.pickle'

# Model Parameters
default_parameters = {
    'enc': {'featureCount': 27,
            'resolution': 0.31096384229403606,
            'size': 62,
            'sparsity': 0.22358451775771981},
    'epochs': 1,
    'ratio': 0.48300309789991624,
    'sp': {'boostStrength': 3.5158429366572728,
           'columnCount': 1369,
           'localAreaDensity': 0.04884751668248205,
           'potentialPct': 0.9076699930632756,
           'potentialRadius': 2363,
           'synPermActiveInc': 0.032426261405406734,
           'synPermConnected': 0.14503910955639598,
           'synPermInactiveDec': 0.009950307527727686},
    'tm': {'activationThreshold': 11,
           'cellsPerColumn': 8,
           'initialPerm': 0.22083745592443638,
           'maxSegmentsPerCell': 161,
           'maxSynapsesPerSegment': 48,
           'minThreshold': 8,
           'newSynapseCount': 18,
           'permanenceDec': 0.0806042024027492,
           'permanenceInc': 0.08121052484196703}
}

# default_parameters = defaultdict(dict)
#
# default_parameters['enc']['featureCount'] = 23
# default_parameters['enc']['resolution'] = 0.3441716670554974
# default_parameters['enc']['size'] = 58
# default_parameters['enc']['sparsity'] = 0.19358046512476695
# default_parameters['epochs'] = 11
# default_parameters['ratio'] = 0.5312116316753737
# default_parameters['sp']['boostStrength'] = 3.297066986627211
# default_parameters['sp']['columnCount'] = 1703
# default_parameters['sp']['localAreaDensity'] = 0.039085079487620675
# default_parameters['sp']['potentialPct'] = 0.7950085867061535
# default_parameters['sp']['potentialRadius'] = 1989
# default_parameters['sp']['synPermActiveInc'] = 0.03910372418550101
# default_parameters['sp']['synPermConnected'] = 0.16883479060489148
# default_parameters['sp']['synPermInactiveDec'] = 0.008965895972154296
# default_parameters['tm']['cellsPerColumn'] = 10
# default_parameters['tm']['activationThreshold'] = 11
# default_parameters['tm']['cellsPerColumn'] = 8
# default_parameters['tm']['initialPerm'] = 0.2378127661016604
# default_parameters['tm']['maxSegmentsPerCell'] = 145
# default_parameters['tm']['maxSynapsesPerSegment'] = 50
# default_parameters['tm']['minThreshold'] = 6
# default_parameters['tm']['newSynapseCount'] = 22
# default_parameters['tm']['permanenceDec'] = 0.05762752507781214
# default_parameters['tm']['permanenceInc'] = 0.09431924505176628

# pprint.pprint(default_parameters)