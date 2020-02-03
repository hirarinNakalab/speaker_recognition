from attrdict import AttrDict
import os

from helpers import Learner
from param import default_parameters
import param



def main(parameters=default_parameters, argv=None, verbose=True):
    parameters = AttrDict(parameters)

    input_path = os.path.join(param.input_dir, "31")
    learner = Learner(
        input_path=input_path,
        setting=parameters,
        unknown="f0001",
        save_threshold=0.8
    )

    print("training epochs: ", parameters.epochs)
    learner.fit(epochs=parameters.epochs)
    f1, cm, report = learner.evaluate()

    print(report)
    print(cm)

    return f1


if __name__ == '__main__':
    main()