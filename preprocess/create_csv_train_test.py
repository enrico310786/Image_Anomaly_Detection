import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    '''
    Script to create the csv files to train and test the model
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_dataset', type=str, help='directory where is stored the mov dataset')

    opt = parser.parse_args()
    dir_dataset = opt.dir_dataset

    df_train = pd.DataFrame(columns=['CLASS', 'LABEL', 'PATH'])
    df_test = pd.DataFrame(columns=['CLASS', 'LABEL', 'PATH'])

    # Iterate in the subdirs of the dataset. Each subdir is a class
    for _, sub_dirs, _ in os.walk(dir_dataset):
        for sub_dir in sub_dirs:
            if sub_dir == "train" or sub_dir == "test":
                path_sub_dir = os.path.join(dir_dataset, sub_dir)
                CHECK_FOLDER = os.path.isdir(path_sub_dir)
                if CHECK_FOLDER:
                    for _, classes, _ in os.walk(path_sub_dir):
                        for classe in classes:
                            path_classe = os.path.join(path_sub_dir, classe)

                            for image in os.listdir(path_classe):
                                path_image = os.path.join(path_classe, image)
                                relative_path = os.path.join(sub_dir, classe, image)

                                if sub_dir == "train":
                                    df_train = df_train.append({'CLASS': classe,
                                                                      'PATH': relative_path,
                                                                      'LABEL': 0}, ignore_index=True)
                                else:
                                    if classe == "good":
                                        df_test = df_test.append({'CLASS': classe,
                                                                        'PATH': relative_path,
                                                                        'LABEL': 0}, ignore_index=True)
                                    else:
                                        df_test = df_test.append({'CLASS': classe,
                                                                        'PATH': relative_path,
                                                                        'LABEL': 1}, ignore_index=True)


    df_train, df_val = train_test_split(df_train, test_size=0.2, random_state=42)
    df_train.to_csv(os.path.join(dir_dataset, "df_train.csv"), index=False)
    df_val.to_csv(os.path.join(dir_dataset, "df_val.csv"), index=False)
    df_test.to_csv(os.path.join(dir_dataset, "df_test.csv"), index=False)

    print("TRAIN DATASET FEATURES")
    print(df_train.info())
    print("")
    print("CLIP DISTRIBUTION BY CLASS")
    print("")
    desc_grouped = df_train[['CLASS']].value_counts()
    print(desc_grouped)
    print("")
    print("VAL DATASET FEATURES")
    print(df_val.info())
    print("")
    print("CLIP DISTRIBUTION BY CLASS")
    print("")
    desc_grouped = df_val[['CLASS']].value_counts()
    print(desc_grouped)
    print("")
    print("TEST DATASET FEATURES")
    print(df_test.info())
    print("")
    print("CLIP DISTRIBUTION BY CLASS")
    print("")
    desc_grouped = df_test[['CLASS']].value_counts()
    print(desc_grouped)
    print("")