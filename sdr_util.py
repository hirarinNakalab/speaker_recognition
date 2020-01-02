from htm.bindings.sdr import SDR

import param

setting = param.default_parameters

def get_encoding(encoder, feature):
    encodings = [encoder.encode(feat) for feat in feature]
    encoding = SDR(setting["enc"]["size"] * setting["enc"]["featureCount"])
    encoding.concatenate(encodings)
    return encoding

