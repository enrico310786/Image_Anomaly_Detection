import os
import cv2
import argparse


def extract_frames(video_path, output_folder, final_size, offset):
    # Open the video
    video = cv2.VideoCapture(video_path)
    success, frame = video.read()
    count = 0

    # Iter to extract the frames
    while success:
        # Crop the frames to the center to obtain a square frame
        min_dim = min(frame.shape[0], frame.shape[1])
        center_x, center_y = frame.shape[1] // 2, frame.shape[0] // 2
        half_dim = min_dim // 2
        #print("min_dim: {} - center_x: {} - center_y: {} - half_dim: {}".format(min_dim, center_x, center_y, half_dim))
        cropped_frame = frame[center_y - half_dim - offset:center_y + half_dim - offset, center_x - half_dim:center_x + half_dim]

        # Resize the frames to final_size
        resized_frame = cv2.resize(cropped_frame, (final_size, final_size))

        frame_path = f"{output_folder}/frame_{count}.jpg"
        cv2.imwrite(frame_path, resized_frame)  # Save the frame
        success, frame = video.read()  # read the next frame
        count += 1

    video.release()
    cv2.destroyAllWindows()
    return count


if __name__ == "__main__":
    '''
    Script take the video of the recordered dataset grouped by classes and tranform them from mov to mp4 extension
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_dataset_video', type=str, help='directory where is stored the video dataset')
    parser.add_argument('--dir_dataset_images', type=str, help='directory where to store the images')
    parser.add_argument('--image_size', type=int, default=256, help='final size for the images')
    parser.add_argument('--offset', type=int, default=15)

    opt = parser.parse_args()
    dir_dataset_video = opt.dir_dataset_video
    dir_dataset_images = opt.dir_dataset_images
    image_size = int(opt.image_size)
    offset = int(opt.offset)

    print("offset: ", offset)

    print("Create the directory: {}".format(dir_dataset_images))
    os.makedirs(dir_dataset_images, exist_ok=True)

    print("----------------------------------------------------------------------")

    # Iterate in the subdirs of the dataset. Each subdir is a category
    for _, categories, _ in os.walk(dir_dataset_video):
        for category in categories:
            path_category = os.path.join(dir_dataset_video, category)
            CHECK_FOLDER = os.path.isdir(path_category)

            if CHECK_FOLDER:
                print("CATEGORY: ", category)

                # create the directory for the images
                path_images_category = os.path.join(dir_dataset_images, category)
                #print("Create the directory: {}".format(path_images_category))
                os.makedirs(path_images_category, exist_ok=True)

                # iter over the video inside the category
                video_category_list = os.listdir(path_category)
                #print("Number of video: {}".format(len(video_category_list)))

                for name_video in video_category_list:

                    if name_video.endswith(".mov"):
                        example = name_video.split(".")[0]
                        path_video = os.path.join(path_category, name_video)

                        # create the directory for the images
                        path_images_example = os.path.join(path_images_category, example)
                        #print("Create the directory: {}".format(path_images_example))
                        os.makedirs(path_images_example, exist_ok=True)

                        number_frames = extract_frames(path_video, path_images_example, image_size, offset)

                        print("Example: {} - Number of images: {}".format(example, number_frames))

            print("----------------------------------------------------------------------")