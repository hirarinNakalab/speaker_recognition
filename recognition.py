import argparse

from helpers import Learner
import param


def main(args, default_parameters=param.default_parameters):
    learner = Learner(
        input_path=param.input_path,
        setting=default_parameters
    )

    print("training epoch: ", args.epoch)
    learner.fit(epoch=args.epoch)
    f1, cm = learner.evaluate()

    print("f1 score: ", f1)
    print(cm)

    return f1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epoch', type=int, default=10,
                        help='Number of epoch to learn, defaults to 10 epochs.')
    args = parser.parse_args()

    main(args)