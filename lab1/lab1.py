import cv2
import numpy as np
from utils import (
        get_log,
        setup_logging,
    )


def recordVideo(nameVideo: str, camera_id: int):
    cap=None
    out=None
    log = get_log("recording")
    try:

        cap = cv2.VideoCapture(camera_id)

        # Default resolutions of the frame are obtained.The default resolutions are system dependent.
        # We convert the resolutions from float to integer.
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))


        # Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
        out = cv2.VideoWriter(nameVideo,cv2.VideoWriter_fourcc('M','J','P','G'), 30, (frame_width,frame_height))

        # stream
        log.info("Start to show web video(id_camera is {camera}) and to save as {name_video}",
            extra={
                "camera": camera_id,
                "name_video": nameVideo,
                }
            )

        while(cap.isOpened()):
          ret, frame = cap.read()
          if ret == True:

            cv2.imshow('Save video',frame)
            out.write(frame)

            # if you want to stop, you have to press q 
            if cv2.waitKey(1) & 0xFF == ord('q'):
              log.info('Stop recording')
              break
          else: 
            break
    except Exception as e:
        log.exception("Error with recording {e}",
                               extra={'e': e})
    finally:
        if cap is not None:
            cap.release()
            log.debug('Close cap')
        if out is not None:    
            out.release()
            log.debug('Save video as {name_video}',
                extra={
                "name_video": nameVideo,
                })
        cv2.destroyAllWindows()
        log.debug('Close window')

def editVideo(openNameVideo: str, saveNameVideo: str):
    cap=None
    out=None
    log = get_log("recording")
    try:

        cap = cv2.VideoCapture(openNameVideo)

        # Default resolutions of the frame are obtained.The default resolutions are system dependent.
        # We convert the resolutions from float to integer.
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))


        # Define the codec and create VideoWriter object.
        out = cv2.VideoWriter(saveNameVideo,cv2.VideoWriter_fourcc('M','J','P','G'), 30, (frame_width,frame_height))

        # stream
        log.info("Start to show edited video ({openNameVideo}) and to save as {saveNameVideo}",
            extra={
                "openNameVideo": openNameVideo,
                "saveNameVideo": saveNameVideo,
                }
            )

        while(cap.isOpened()):
          ret, frame = cap.read()
          if ret == True:

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray= cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

            cv2.rectangle(gray, (100, 100), (200, 200), (0, 255, 255), 10)
            cv2.line(gray, (60, 20), (400, 200), (0, 100, 255), 5)


            cv2.imshow('Edited video',gray)
            out.write(gray)

            # if you want to stop, you have to press q 
            if cv2.waitKey(1) & 0xFF == ord('q'):
              log.info('Stop editing')
              break
          else: 
            break
    except Exception as e:
        log.exception("Error with editing {e}",
                               extra={'e': e})
    finally:
        if cap is not None:
            cap.release()
            log.debug('Close cap')
        if out is not None:    
            out.release()
            log.debug('Save video as {name_video}',
                extra={
                "name_video": saveNameVideo,
                })
        cv2.destroyAllWindows()
        log.debug('Close window')



def main():
    log = get_log('lab1')
    log.info("start video recording")
    recordVideo("outvideo.avi", 0)
    editVideo("outvideo.avi","outvideo2.avi")


if __name__ == '__main__':
    setup_logging("lab1")
    main()


