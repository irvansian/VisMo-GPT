# import cv2
# import os
# import glob
#
# def extract_frames(video_path, start_second, end_second, output_dir):
#     os.makedirs(output_dir, exist_ok=True)
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         print(f"Failed to open video: {video_path}")
#         cap.release()
#         return None  # Return None if video file could not be opened
#     fps = int(cap.get(cv2.CAP_PROP_FPS))
#     start_frame = start_second * fps
#     end_frame = end_second * fps
#     cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
#
#     output_frame_number = 0
#     frame_number = start_frame
#     while cap.isOpened() and frame_number <= end_frame:
#         ret, frame = cap.read()
#         if ret:
#             frame_filename = os.path.join(output_dir, f'{output_frame_number:04d}.png')
#             cv2.imwrite(frame_filename, frame)
#             frame_number += 1
#             output_frame_number += 1
#         else:
#             break
#     cap.release()
#     return fps  # Return the frames per second
#
# def frames_to_video(input_path, output_path, fps):
#     frame_files = sorted(glob.glob(os.path.join(input_path, '*.png')),
#                          key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
#
#     frame = cv2.imread(frame_files[0])
#     h, w, layers = frame.shape
#     size = (w, h)
#     out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
#
#     for frame_file in frame_files:
#         img = cv2.imread(frame_file)
#         out.write(img)
#
#     out.release()
#
# # Usage
# base_dir = os.path.dirname(os.path.abspath(__file__))  # Gets the directory of the current script
# video_file = os.path.join(base_dir, 'video', 'nike.mp4')
# start_time = 10  # Start at 10 seconds
# end_time = 15    # End at 15 seconds
# output_directory = os.path.join(base_dir, 'video', 'frames_output', 'nike')
# output_vid_directory = os.path.join(base_dir, 'video', 'frames_output', 'nike', 'processed_nike.mp4')
#
# if __name__ == "__main__":
#     fps = extract_frames(video_file, start_time, end_time, output_directory)
#     if fps:  # Proceed only if fps is not None (i.e., the video file was successfully opened)
#         frames_to_video(output_directory, output_vid_directory, fps)
