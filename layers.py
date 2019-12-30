from htm.bindings.algorithms import SpatialPooler
from htm.bindings.algorithms import TemporalMemory
from htm.bindings.algorithms import Classifier
from htm.encoders.rdse import RDSE, RDSE_Parameters
from htm.bindings.sdr import SDR

import param


def create_encoder(width):
    setting = param.parameters
    scalarEncoderParams = RDSE_Parameters()
    scalarEncoderParams.size = width
    scalarEncoderParams.sparsity = setting["enc"]["sparsity"]
    scalarEncoderParams.resolution = setting["enc"]["resolution"]
    scalarEncoder = RDSE(scalarEncoderParams)
    return scalarEncoder

def create_classifier():
    return Classifier()

class Layer:

    def __init__(self, din=(10, 10), dout=(10, 10), temporal=True, param_dict=param.parameters):
        self.input_shape = din
        self.output_shape = dout
        self.temporal = temporal
        self.param = dict(param_dict)
        self.sp = SpatialPooler()
        self.tm = TemporalMemory() if temporal else None

    def compile(self):
        spParams = self.param["sp"]
        self.sp = SpatialPooler(
            inputDimensions=self.input_shape,
            columnDimensions=self.output_shape,
            potentialPct=spParams['potentialPct'],
            potentialRadius=int(self.input_shape[0] * 3 / 8),
            globalInhibition=spParams['globalInhibition'],
            localAreaDensity=spParams['localAreaDensity'],
            synPermInactiveDec=spParams['synPermInactiveDec'],
            synPermActiveInc=spParams['synPermActiveInc'],
            synPermConnected=spParams['synPermConnected'],
            boostStrength=spParams['boostStrength'],
            wrapAround=spParams['wrapAround'],
        )
        if self.temporal:
            tmParams = self.param["tm"]
            self.tm = TemporalMemory(
                columnDimensions=self.output_shape,
                cellsPerColumn=tmParams["cellsPerColumn"],
                activationThreshold=tmParams["activationThreshold"],
                initialPermanence=tmParams["initialPermanence"],
                connectedPermanence=tmParams["connectedPermanence"],
                minThreshold=tmParams["minThreshold"],
                maxNewSynapseCount=tmParams["maxNewSynapseCount"],
                permanenceIncrement=tmParams["permanenceIncrement"],
                permanenceDecrement=tmParams["permanenceDecrement"],
                predictedSegmentDecrement=tmParams['predictedSegmentDecrement'],
                maxSegmentsPerCell=tmParams["maxSegmentsPerCell"],
                maxSynapsesPerSegment=tmParams["maxSynapsesPerSegment"]
            )

    def predict(self, encoding, learn=True):
        activeColumns = SDR(self.sp.getColumnDimensions())
        self.sp.compute(encoding, learn, activeColumns)

        predictedColumns = None
        if self.temporal:
            self.tm.compute(activeColumns, learn)
            self.tm.activateDendrites(learn)
            predictedColumnIndices = {self.tm.columnForCell(i)
                                      for i in self.tm.getPredictiveCells().sparse}
            predictedColumns = SDR(self.sp.getColumnDimensions())
            predictedColumns.sparse = list(predictedColumnIndices)
        return activeColumns, predictedColumns

    def anomaly(self):
        return float(self.tm.anomaly) if self.temporal else None

    def reset(self):
        self.tm.reset() if self.temporal else None

    def save(self, i):
        print('Saving Model...')
        print(str(self.sp))
        self.sp.saveToFile(param.sp_model.format(i))
        if self.temporal:
            print(str(self.tm))
            self.tm.saveToFile(param.tm_model.format(i))

    def load(self, i):
        print('Loading Model...')
        self.sp.loadFromFile(param.sp_model.format(i))
        print(str(self.sp))
        if self.temporal:
            self.tm.loadFromFile(param.tm_model.format(i))
            print(str(self.tm))


class Region:

    def __init__(self, *args):
        self.regions = [region for region in args if isinstance(region, Layer)]
        self.reg_num = len(self.regions)

    def compile(self):
        for i, reg in enumerate(range(self.reg_num)):
            self.regions[i].compile()

    def save(self):
        for i, reg in enumerate(range(self.reg_num)):
            self.regions[i].save(i)

    def load(self):
        for i, reg in enumerate(range(self.reg_num)):
            self.regions[i].load(i)

    def forward(self, x, learn=True):
        outputs = []
        for i, reg in enumerate(range(self.reg_num)):
            act, x = self.regions[i].predict(x, learn)
            outputs.append((act, x))
        return outputs

    def anomaly(self):
        return float(self.regions[-1].tm.anomaly) if self.regions[-1].temporal else None

    def reset(self):
        for i, reg in enumerate(range(self.reg_num)):
            self.regions[i].reset()