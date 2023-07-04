import os


TOTAL_VIDEO_CHUNCK = 49
BITRATE_LEVELS = 6
VIDEO_PATH = '../video_server/'
VIDEO_FOLDER = 'video'


for bitrate in range(BITRATE_LEVELS):
	with open('video_size_' + str(bitrate), 'w') as f:
		for chunk_num in range(1, TOTAL_VIDEO_CHUNCK+1):
			video_chunk_path = VIDEO_PATH + \
							   VIDEO_FOLDER + \
							   str(BITRATE_LEVELS - bitrate) + \
							   '/' + \
							   str(chunk_num) + \
							   '.m4s'
			chunk_size = os.path.getsize(video_chunk_path)
			f.write(str(chunk_size) + '\n')
