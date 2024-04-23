import cv2
import os

def extract_frames(video_path, output_folder, final_size):
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
        cropped_frame = frame[center_y - half_dim:center_y + half_dim, center_x - half_dim:center_x + half_dim]

        # Resize the frames to final_size
        resized_frame = cv2.resize(cropped_frame, (final_size, final_size))

        frame_path = f"{output_folder}/frame_{count}.jpg"
        cv2.imwrite(frame_path, resized_frame)  # Save the frame
        success, frame = video.read()  # read the next frame
        count += 1

    print("Total frames: {}".format(count))

    video.release()
    cv2.destroyAllWindows()


video_path = '/home/enrico/Dataset/images_anomaly/video_lego/one_up/one_up.mov'
output_folder = '/home/enrico/Dataset/images_anomaly/test_images_lego'
final_size = 1024
os.makedirs(output_folder, exist_ok=True)
extract_frames(video_path, output_folder, final_size)