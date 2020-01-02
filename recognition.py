import random

from layers import create_encoder, create_model, create_clf
from helpers import create_speakers_dict, create_dataset, get_answer_and_prediction, get_f1_and_cm, get_length_dict, get_phase_whole_data, train
import param



def main(default_parameters=param.default_parameters):
    speakers = 'unk m0001 f0002'
    speakers_dict = create_speakers_dict(speakers)

    dataset = create_dataset(param.input_file, speakers_dict)
    encoder = create_encoder()

    length_dict = get_length_dict(dataset, speakers_dict)

    models = {}
    for phase in ['train', 'test']:
        print("====={}ing phase=====\n".format(phase))
        length = length_dict[phase]

        if phase == "train":
            for speaker in speakers_dict.keys():
                print("="*30 + "model of ", speaker, "="*30+"\n")

                model = create_model()
                models[speaker] = model
                model.train()

                train_data = dataset[phase][speaker][:length]
                num_epochs = 10
                for epoch in range(num_epochs):
                    print("epoch {}".format(epoch))
                    train(random.shuffle(train_data), encoder, model)

                print("{}ing data count: {}".format(phase, len(train_data)), end='\n\n')

        elif phase == "test":
            clf = create_clf(model_dict=models, speakers_dict=speakers_dict)
            train_data = get_phase_whole_data("train", dataset, speakers_dict, length_dict)
            clf.optimize(train_data)

            test_data = get_phase_whole_data(phase, dataset, speakers_dict, length_dict)
            ans, pred = get_answer_and_prediction(train_data, speakers_dict, clf)

            f1, cm = get_f1_and_cm(ans, pred)

            print("{}ing data count: {}".format(phase, len(test_data)), end='\n\n')

    return f1



if __name__ == '__main__':
    main()