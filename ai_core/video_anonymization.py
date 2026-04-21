class VideoAnonymization:
    def __init__(self, video_io, face_detector, face_tracker, face_anonymizer):
        self.video_io = video_io
        self.face_detector = face_detector
        self.face_tracker = face_tracker
        self.face_anonymizer = face_anonymizer
        
    def anonymize_video_without_model(self): # blur, pixelate, etc.
        pass
    def anonymize_video_with_model(self): # face swapping, etc.
        pass