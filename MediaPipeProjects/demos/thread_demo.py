import cv2
from CountsPerSec import CountsPerSec
from MediaPipeProjects.FULLBODY_Data.VideoGet import VideoGet
# from VideoShow import VideoShow

def putIterationsPerSec(frame, iterations_per_sec):
    """
    Add iterations per second text to lower-left corner of a frame.
    """

    cv2.putText(frame, "{:.0f} iterations/sec".format(iterations_per_sec),
        (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255))
    return frame


def noThreading (source=0):

    cap = cv2.VideoCapture(source)
    cps = CountsPerSec().start()

    while True:
        (grabbed, frame) = cap.read()
        if not grabbed or cv2.waitKey(1) == ord("q"):
            break

        frame = putIterationsPerSec(frame, cps.countsPerSec())
        cv2.imshow("Video", frame)
        cps.increment()


def threadVideoGet(source=""):
    """
    Dedicated thread for grabbing video frames with VideoGet object.
    Main thread shows video frames.
    """

    video_getter = VideoGet(source).start()
    cps = CountsPerSec().start()

    while True:
        if (cv2.waitKey(1) == ord("q")) or video_getter.stopped:
            video_getter.stop()
            break

        frame = video_getter.frame
        frame = putIterationsPerSec(frame, cps.countsPerSec())
        cv2.imshow("Video", frame)
        cps.increment()


threadVideoGet("sub-002_ses-002_task-hokey_cokey_run-005_video.mp4")