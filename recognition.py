from attrdict import AttrDict
import os

from helpers import Learner
from param import default_parameters
import param



def main(parameters=default_parameters, argv=None, verbose=True):
    parameters = AttrDict(parameters)
    args = AttrDict(dict(speech="31", ratio=0.8, unknown="f0001", epochs=1))

    input_path = os.path.join(param.input_dir, str(args.speech))
    learner = Learner(
        input_path=input_path,
        setting=parameters,
        split_ratio=args.ratio,
        unknown=args.unknown
    )

    print("training epochs: ", args.epochs)
    learner.fit(epochs=args.epochs)
    f1, cm, report = learner.evaluate()

    print(report)
    print(cm)

    return f1


if __name__ == '__main__':
    main()