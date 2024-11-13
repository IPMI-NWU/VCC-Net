from a_Data.dataset_COVID import Dataset_COVID
from a_Data.dataset_MIMIC import Dataset_MIMIC
from a_Data.dataset_SIIM import Dataset_SIIM
from a_Data.dataset_TB import Dataset_TB


def build_dataset(dataset_name, phase, img_size):
    dataset = None

    if dataset_name == 'SIIM':
        if phase == 'train':
            dataset = Dataset_SIIM('../a_Data/SIIM-ACR/train/', '../a_Data/SIIM-ACR/siim.csv', 'train', img_size)
        else:
            dataset = Dataset_SIIM('../a_Data/SIIM-ACR/test/', '../a_Data/SIIM-ACR/siim.csv', 'test', img_size)

    if dataset_name == 'MIMIC':
        if phase == 'train':
            dataset = Dataset_MIMIC('../a_Data/MIMIC-Gaze/train/', '../a_Data/MIMIC-Gaze/mimic_part.csv', 'train', img_size)
        else:
            dataset = Dataset_MIMIC('../a_Data/MIMIC-Gaze/test/', '../a_Data/MIMIC-Gaze/mimic_part.csv', 'test', img_size)

    if dataset_name == 'TB':
        if phase == 'train':
            dataset = Dataset_TB('../a_Data/TB-Mouse/train/', '../a_Data/TB-Mouse/MouseData.csv', 'train', img_size)
        else:
            dataset = Dataset_TB('../a_Data/TB-Mouse/test/', '../a_Data/TB-Mouse/test.csv', 'test', img_size)

    if dataset_name == 'COVID':
        if phase == 'train':
            dataset = Dataset_COVID('../a_Data/COVID-QU-Ex/', '../a_Data/COVID-QU-Ex/train_test.csv', 'train', img_size)
        else:
            dataset = Dataset_COVID('../a_Data/COVID-QU-Ex/', '../a_Data/COVID-QU-Ex/train_test.csv', 'test', img_size)

    return dataset

if __name__ == '__main__':
    dataset = build_dataset('SIIM', 'train', 224)
    dataset = build_dataset('SIIM', 'test', 224)

    dataset = build_dataset('MIMIC', 'train', 224)
    dataset = build_dataset('MIMIC', 'test', 224)

    dataset = build_dataset('TB', 'train', 224)
    dataset = build_dataset('TB', 'test', 224)

    dataset = build_dataset('COVID', 'train', 224)
    dataset = build_dataset('COVID', 'test', 224)