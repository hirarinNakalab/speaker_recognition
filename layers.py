from attrdict import AttrDict
from htm.bindings.algorithms import SpatialPooler
from htm.bindings.algorithms import TemporalMemory
from htm.bindings.sdr import SDR

import param

class Layer:
    def __init__(self, din=(10, 10), dout=(10, 10), temporal=True, setting=param.default_parameters):
        self.input_shape = din
        self.output_shape = dout
        self.temporal = temporal
        self.learn = True
        self.setting = AttrDict(setting)
        self.sp = SpatialPooler()
        self.tm = TemporalMemory() if temporal else None

    def compile(self):
        spParams = self.setting("sp")
        self.sp = SpatialPooler(
            inputDimensions=self.input_shape,
            columnDimensions=self.output_shape,
            potentialPct=spParams.potentialPct,
            potentialRadius=spParams.potentialRadius,
            globalInhibition=True if len(self.output_shape) == 1 else False,
            localAreaDensity=spParams.localAreaDensity,
            synPermInactiveDec=spParams.synPermInactiveDec,
            synPermActiveInc=spParams.synPermActiveInc,
            synPermConnected=spParams.synPermConnected,
            boostStrength=spParams.boostStrength,
            wrapAround=True,
        )
        if self.temporal:
            tmParams = self.setting("tm")
            self.tm = TemporalMemory(
                columnDimensions=self.output_shape,
                cellsPerColumn=tmParams.cellsPerColumn,
                activationThreshold=tmParams.activationThreshold,
                initialPermanence=tmParams.initialPerm,
                connectedPermanence=spParams.synPermConnected,
                minThreshold=tmParams.minThreshold,
                maxNewSynapseCount=tmParams.newSynapseCount,
                permanenceIncrement=tmParams.permanenceInc,
                permanenceDecrement=tmParams.permanenceDec,
                predictedSegmentDecrement=0.0,
                maxSegmentsPerCell=tmParams.maxSegmentsPerCell,
                maxSynapsesPerSegment=tmParams.maxSynapsesPerSegment
            )

    def forward(self, encoding):
        activeColumns = SDR(self.sp.getColumnDimensions())
        self.sp.compute(encoding, self.learn, activeColumns)

        predictedColumns = None
        if self.temporal:
            self.tm.compute(activeColumns, self.learn)
            self.tm.activateDendrites(self.learn)
            predictedColumnIndices = {self.tm.columnForCell(i)
                                      for i in self.tm.getPredictiveCells().sparse}
            predictedColumns = SDR(self.sp.getColumnDimensions())
            predictedColumns.sparse = list(predictedColumnIndices)
        return activeColumns, predictedColumns

    def train(self):
        self.learn = True

    def eval(self):
        self.learn = False

    def anomaly(self):
        return float(self.tm.anomaly) if self.temporal else None

    def reset(self):
        if self.temporal:
            self.tm.reset()

    def save(self, path):
        print('Saving Model...')
        print(str(self.sp))

        self.sp.saveToFile(param.sp_model.format(path))
        if self.temporal:
            print(str(self.tm))
            self.tm.saveToFile(param.tm_model.format(path))

    def load(self, path):
        print('Loading Model...')
        self.sp.loadFromFile(param.sp_model.format(path))
        print(str(self.sp))
        if self.temporal:
            self.tm.loadFromFile(param.tm_model.format(path))
            print(str(self.tm))


class Unknown:
    def __init__(self, threshold=1.0):
        self.threshold = threshold

    def compile(self):
        pass

    def forward(self, encoding):
        return None, None

    def train(self):
        pass

    def eval(self):
        pass

    def anomaly(self):
        return self.threshold

    def reset(self):
        pass

    def save(self, filename):
        pass

# class Region:
#     def __init__(self, *args):
#         self.units = [arg for arg in args if isinstance(arg, Layer)]
#         self.units_num = len(self.units)
#         self.learn = True
#
#     def compile(self):
#         for i, unit in enumerate(range(self.units_num)):
#             self.units[i].compile()
#
#     def save(self):
#         for i, unit in enumerate(range(self.units_num)):
#             self.units[i].save(i)
#
#     def load(self):
#         for i, unit in enumerate(range(self.units_num)):
#             self.units[i].load(i)
#
#     def train(self):
#         for i, unit in enumerate(range(self.units_num)):
#             self.units[i].train()
#
#     def eval(self):
#         for i, unit in enumerate(range(self.units_num)):
#             self.units[i].eval()
#
#     def forward(self, x):
#         outputs = []
#         for i, unit in enumerate(range(self.units_num)):
#             act, x = self.units[i].forward(x)
#             outputs.append((act, x))
#         return outputs
#
#     def anomaly(self):
#         return float(self.units[-1].tm.anomaly) if self.units[-1].temporal else None
#
#     def reset(self):
#         for i, unit in enumerate(range(self.units_num)):
#             self.units[i].reset()