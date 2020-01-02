from sklearn.metrics import confusion_matrix, f1_score
import random

from layers import create_encoder, create_model, create_clf
from helpers import create_speakers_dict, create_dataset, get_speaker_idx, get_length_dict, experiment
import param



def main(default_parameters=param.default_parameters):
    speakers = 'm0001 f0002'
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
                    experiment(random.shuffle(train_data), encoder, model)

                print("{}ing data count: {}".format(phase, len(train_data)), end='\n\n')

        elif phase == "test":
            clf = create_clf(model_dict=models, speakers_dict=speakers_dict)
            train_data = [data
                        for speaker in speakers_dict.keys()
                        for data in dataset["train"][speaker][:length_dict["train"]]]
            clf.optimize(train_data)

            test_data = [data
                         for speaker in speakers_dict.keys()
                         for data in dataset[phase][speaker][:length]]

            answer = [get_speaker_idx(speakers_dict, wav) for wav in test_data]
            prediction = [clf.predict(wav) for wav in test_data]

            cm = confusion_matrix(answer, prediction)
            f1 = f1_score(answer, prediction)

            print("{}ing data count: {}".format(phase, len(test_data)), end='\n\n')

    return f1



if __name__ == '__main__':
    main()