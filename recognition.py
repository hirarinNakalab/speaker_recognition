import os
import argparse

from helpers import Learner
import param


def main(args, default_parameters=param.default_parameters):
    input_path = os.path.join(param.input_dir, str(args.speech))
    learner = Learner(
        input_path=input_path,
        setting=default_parameters
    )

    print("training epoch: ", args.epoch)
    learner.fit(epoch=args.epoch)
    f1, cm = learner.evaluate()

    print(cm)
    print(f1)

    return f1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epoch', type=int, default=5,
                        help='Number of epoch to learn, defaults to 5 epochs.')
    parser.add_argument('-s', '--speech', type=int, default=31,
                        help='The number of speech to learn, defaults to #31')
    args = parser.parse_args()

    main(args)