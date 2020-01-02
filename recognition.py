from layers import create_encoder, create_model
from helpers import create_speakers_dict, create_dataset, get_length_dict, experiment
import param



def main():
    speakers = 'm0001 f0002'
    speakers_dict = create_speakers_dict(speakers)

    dataset = create_dataset(param.input_file, speakers_dict)
    encoder = create_encoder()

    length_dict = get_length_dict(dataset, speakers_dict)

    models = {}
    for phase in ['train', 'test']:
        print("====={}ing phase=====\n".format(phase))
        length = length_dict[phase]

        for speaker in speakers_dict.keys():
            print("="*30)
            print("model of ", speaker)
            print("="*30+"\n")

            if phase == "train":
                model = create_model()
                models[speaker] = model
                model.train()

                wav_data = dataset[phase][speaker][:length]
            else:
                model = models[speaker]
                model.eval()

                wav_data = [data
                             for speaker in speakers_dict.keys()
                             for data in dataset[phase][speaker][:length]]

            experiment(wav_data, encoder, model)
            print("{}ing data count: {}".format(phase, len(wav_data)), end='\n\n')


if __name__ == '__main__':
    main()