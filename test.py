from ai_core.video_io.video_io import VideoIO

if __name__ == "__main__":
    video_io = VideoIO()
    metadata1 = video_io.get_video_metadata("test_videos/test1.mp4")
    metadata2 = video_io.get_video_metadata("test_videos/test2.mp4")
    metadata3 = video_io.get_video_metadata("test_videos/test3.mp4")
    metadata4 = video_io.get_video_metadata("test_videos/test4.mp4")
    
    print(metadata1)
    print(metadata2)
    print(metadata3)
    print(metadata4)
    
