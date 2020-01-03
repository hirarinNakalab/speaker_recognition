from helpers import Learner
import param


def main(default_parameters=param.default_parameters):
    learner = Learner(path=param.input_path, setting=default_parameters)

    print("=====training phase=====\n")
    learner.fit(epoch=10)

    print("=====training phase=====\n")
    f1 = learner.evaluate()

    print("f1 score: ", f1)
    return f1


if __name__ == '__main__':
    main()