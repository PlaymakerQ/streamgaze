import json, os, re

qa = "dataset/qa/past_scene_recall.json"
root = "dataset/videos/gaze_viz_video"

data = json.load(open(qa))
missing = []

for item in data:
    s = str(item)
    vids = re.findall(r'[\w\-]+\.mp4', s)
    for v in vids:
        if not os.path.exists(os.path.join(root, v)):
            missing.append(v)

print("missing:", len(set(missing)))
print(list(sorted(set(missing)))[:20])