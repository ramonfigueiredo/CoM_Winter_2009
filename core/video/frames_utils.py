import cv2

from core.video.video_processing.video_processing_utils import get_rotation_options


# get_resized_image, show_frame_id, get_video_frame_rotation_angle
def get_resized_image(frame, scale_percent=60):
    # scale_percent = percent of original size
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)

    # resize image
    return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)


def show_frame_id(frame, frame_id):
    cv2.putText(frame,
                'Frame {}'.format(frame_id), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), thickness=2)


def get_video_frame_rotation_angle(video):
    ROTATION_90_DEGREES_CLOCKWISE, ROTATION_180_DEGREES_CLOCKWISE, ROTATION_270_DEGREES_CLOCKWISE = \
        get_rotation_options()

    if video.uploaded_video_rotation_option:
        if video.get_uploaded_video_rotation_option_display() == ROTATION_90_DEGREES_CLOCKWISE:
            frame_rotation_angle = cv2.ROTATE_90_CLOCKWISE

        elif video.get_uploaded_video_rotation_option_display() == ROTATION_180_DEGREES_CLOCKWISE:
            frame_rotation_angle = cv2.ROTATE_180

        elif video.get_uploaded_video_rotation_option_display() == ROTATION_270_DEGREES_CLOCKWISE:
            frame_rotation_angle = cv2.ROTATE_90_COUNTERCLOCKWISE

        return frame_rotation_angle
    else:
        return None
