import numpy as np

from htm.bindings.algorithms import SpatialPooler
from htm.bindings.algorithms import TemporalMemory
from htm.encoders.rdse import RDSE, RDSE_Parameters
from htm.bindings.sdr import SDR

from helpers import experiment, get_answer_and_prediction, get_f1_and_cm
import param

setting = param.default_parameters

def create_encoder():
    print("creating encoder...")
    print(setting["enc"])
    scalarEncoderParams = RDSE_Parameters()
    scalarEncoderParams.size = setting["enc"]["size"]
    scalarEncoderParams.sparsity = setting["enc"]["sparsity"]
    scalarEncoderParams.resolution = setting["enc"]["resolution"]
    scalarEncoder = RDSE(scalarEncoderParams)
    print()
    return scalarEncoder

def create_model():
    print("creating model...")
    print(setting["sp"])
    print(setting["tm"])
    model = Layer(
        din=(setting["enc"]["size"] * setting["enc"]["featureCount"],),
        dout=(setting["sp"]["columnCount"],)
    )
    model.compile()
    print()
    return model

def create_clf(model_dict, speakers_dict):
    return OVRClassifier(model_dict, speakers_dict)



class OVRClassifier:
    def __init__(self, model_dict, speakers_dict, encoder):
        self.threshold = 0
        self.model_dict = model_dict
        self.speakers_dict = speakers_dict
        self.encoder = encoder

    def optimize(self, train_data, encoder):
        ths = np.linspace(0, 1, 1000)
        for th in ths:
            self.threshold = th
            ans, pred = get_answer_and_prediction(train_data, self.speakers_dict, self)
            results = {th: get_f1_and_cm(ans, pred)[0]}

        results_sorted = sorted(results.items(), key=lambda x: x[1], reverse=True)
        self.threshold = float(results_sorted[0].key())

    def predict(self, data):
        anomalies = {}
        for speaker in self.speakers_dict.keys():
            model = self.model_dict[speaker]
            model.eval()
            anomalies[speaker] = experiment(data, self.encoder, model)
        anom_sorted = sorted(anomalies.items(), key=lambda x: x[1], reverse=True)
        return self.speakers_dict["unk"] if all(anomalies > self.threshold) else self.speakers_dict[
            anom_sorted[0].key()]


class Layer:
    def __init__(self, din=(10, 10), dout=(10, 10), temporal=True, param_dict=param.default_parameters):
        self.input_shape = din
        self.output_shape = dout
        self.temporal = temporal
        self.learn = True
        self.param = dict(param_dict)
        self.sp = SpatialPooler()
        self.tm = TemporalMemory() if temporal else None

    def compile(self):
        spParams = self.param["sp"]
        self.sp = SpatialPooler(
            inputDimensions=self.input_shape,
            columnDimensions=self.output_shape,
            potentialPct=spParams['potentialPct'],
            potentialRadius=self.input_shape[0],
            globalInhibition=True if len(self.output_shape)==1 else False,
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
                initialPermanence=tmParams["initialPerm"],
                connectedPermanence=spParams["synPermConnected"],
                minThreshold=tmParams["minThreshold"],
                maxNewSynapseCount=tmParams["newSynapseCount"],
                permanenceIncrement=tmParams["permanenceInc"],
                permanenceDecrement=tmParams["permanenceDec"],
                predictedSegmentDecrement=0.0,
                maxSegmentsPerCell=tmParams["maxSegmentsPerCell"],
                maxSynapsesPerSegment=tmParams["maxSynapsesPerSegment"]
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
        self.units = [arg for arg in args if isinstance(arg, Layer)]
        self.units_num = len(self.units)
        self.learn = True

    def compile(self):
        for i, unit in enumerate(range(self.units_num)):
            self.units[i].compile()

    def save(self):
        for i, unit in enumerate(range(self.units_num)):
            self.units[i].save(i)

    def load(self):
        for i, unit in enumerate(range(self.units_num)):
            self.units[i].load(i)

    def train(self):
        for i, unit in enumerate(range(self.units_num)):
            self.units[i].train()

    def eval(self):
        for i, unit in enumerate(range(self.units_num)):
            self.units[i].eval()

    def forward(self, x):
        outputs = []
        for i, unit in enumerate(range(self.units_num)):
            act, x = self.units[i].forward(x)
            outputs.append((act, x))
        return outputs

    def anomaly(self):
        return float(self.units[-1].tm.anomaly) if self.units[-1].temporal else None

    def reset(self):
        for i, unit in enumerate(range(self.units_num)):
            self.units[i].reset()