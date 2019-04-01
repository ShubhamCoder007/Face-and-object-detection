# Face-and-motion-detection
Basic face, eye and smile detection implemented as GUI in python. One option supports uploading picture and then
processing the face, eyes or smile respectively according to the selected option.
Also a seperate realtime option of the webcam is added where the picture from the webcam is taken and then the
face is processed.
Video processing from the webcam is present where face is detected and smile could be detected as well.
Success message is displayed if the file is processed without any error. In case there is any error then the error
message is displayed. Video section would also generate the number of video frames generated.

Motion detection would detect object when it enters the webcam frame. The time when the object appeared and when it
exits is generated as well.

haarcascade object acts as a reference for the detection of the face, eye, smile; with respect to which we find the
aforementioned characteristics.
