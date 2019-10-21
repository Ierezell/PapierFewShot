import os
import glob

global_video_path = 'dataset/mp4/'
CONCAT_VIDEO_NAME = "output.mp4"

person_list = glob.glob(f"{global_video_path}/*")

for person in person_list:
    context_list = glob.glob(f"{person}/*")
    for context in context_list:
        path = os.path.join(context, "videos.txt")
        if os.path.exists(path):
            os.remove(path)
        videos_list = glob.glob(f"{context}/0*")
        res = []
        if len(videos_list) != 0:
            for v in videos_list:
                video_name = v.split("/")[-1]
                res.append("file " + "'" + video_name + "'\n")
            res.sort()
            with open(path, "w") as myfile:
                myfile.write("".join(res))
            try:
                output_path = os.path.join(context, CONCAT_VIDEO_NAME)
                command = "ffmpeg -f concat -i " + path + " -c copy " + output_path
                os.system(command)
                os.remove(path)
                if os.path.exists(output_path):
                    remove = "rm -rf " + context + "/0*.mp4"
                    os.system(remove)
            except:
                print("Error in " + context)
