import os
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_dataset', type=str)

    opt = parser.parse_args()
    dir_dataset = opt.dir_dataset

    # Iterate in the subdirs of the dataset. Each subdir is a category
    for _, categories, _ in os.walk(dir_dataset):
        for category in categories:
            path_category = os.path.join(dir_dataset, category)
            CHECK_FOLDER = os.path.isdir(path_category)

            if CHECK_FOLDER:
                print("CATEGORY: ", category)

                # iterate over the subdir
                for _, classes, _ in os.walk(path_category):
                    for classe in classes:
                        if classe == "90_DEG":
                            path_classe = os.path.join(path_category, classe)
                            number_images = len(os.listdir(path_classe))
                            print("Normal 90_DEG - Number of images: {}".format(number_images))
                        elif classe == "abnormal":
                            path_classe = os.path.join(path_category, classe)
                            #iterate over subclasses
                            for _, subclasses, _ in os.walk(path_classe):
                                for subclasse in subclasses:
                                    path_subclasse = os.path.join(path_classe, subclasse)
                                    number_images = len(os.listdir(path_subclasse))

                                    if subclasse == "30_DEG":
                                        print("Abnormal 30_DEG - Number of images: {}".format(number_images))
                                    elif subclasse == "60_DEG":
                                        print("Abnormal 60_DEG - Number of images: {}".format(number_images))
                                    elif subclasse == "45_DEG":
                                        print("Abnormal 45_DEG - Number of images: {}".format(number_images))

                print("------------------------------------------------------------------------")