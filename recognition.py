import os
import argparse

from helpers import Learner
import param


def main(args, default_parameters=param.default_parameters):
    input_path = os.path.join(param.input_dir, str(args.speech))
    learner = Learner(
        input_path=input_path,
        setting=default_parameters,
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
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', type=int, default=1,
                        help='Number of epoch to learn, defaults to 5 epochs.')
    parser.add_argument('-s', '--speech', type=int, default=31,
                        help='The number of speech to learn, defaults to 31')
    parser.add_argument('-r', '--ratio', type=float, default=0.8,
                        help='The ratio of train/test split, defaults to 0.8')
    parser.add_argument('-u', '--unknown', type=str, default="m0006",
                        help='The ratio of train/test split, defaults to m0006')
    args = parser.parse_args()

    main(args)