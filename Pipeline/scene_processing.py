from scenedetect import detect, ContentDetector

class SceneDetector() :

    def detect_scenes(self, file, thresh) :
        scene_list = detect(file, ContentDetector(threshold=thresh), start_in_scene=True)
        begin_frames = [scene[0].get_frames() for scene in scene_list]
        return(begin_frames)