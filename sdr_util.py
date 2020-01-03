from htm.bindings.sdr import SDR


def get_encoding(encoder, feature, setting):
    encodings = [encoder.encode(feat) for feat in feature]
    encoding = SDR(setting["enc"]["size"] * setting["enc"]["featureCount"])
    encoding.concatenate(encodings)
    return encoding