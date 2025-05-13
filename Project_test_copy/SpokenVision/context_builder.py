import numpy as np
import cv2
from collections import defaultdict

class ContextBuilder:
    def __init__(self, proximity_threshold=0.2, depth_cutoffs=[0.1, 0.25, 0.4, 0.6, 0.75, 0.9]):
        self.proximity_threshold = proximity_threshold
        self.depth_cutoffs = depth_cutoffs
        self.previous_objects = []
        self.new_objects = []
        self.frame_count = 0

    def process_frame_data(self, detection_results, depth_map, segmentation_map, caption):
        self.frame_count += 1
        objects_with_positions = self._process_objects(detection_results, depth_map)
        spatial_relationships = self._analyze_spatial_relationships(objects_with_positions)
        scene_description = self._generate_scene_description(objects_with_positions, spatial_relationships, caption)
        return scene_description

    def _process_objects(self, detection_results, depth_map):
        objects = []
        for box, label, score in zip(detection_results["boxes"], detection_results["labels"], detection_results["scores"]):
            x1, y1, x2, y2 = [int(coord) for coord in box]
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            width = x2 - x1
            height = y2 - y1
            size = width * height
            try:
                x1_valid = max(0, min(x1, depth_map.shape[1] - 1))
                x2_valid = max(0, min(x2, depth_map.shape[1] - 1))
                y1_valid = max(0, min(y1, depth_map.shape[0] - 1))
                y2_valid = max(0, min(y2, depth_map.shape[0] - 1))
                object_depth_region = depth_map[y1_valid:y2_valid, x1_valid:x2_valid]
                avg_depth = object_depth_region.mean() / 255.0 if object_depth_region.size > 0 else 0.5
            except Exception:
                avg_depth = 0.5

            depth_categories = [
                "very far (barely visible)",
                "far (on the horizon)",
                "medium-far (across the room)",
                "medium (about halfway into the room)",
                "medium-close (a few feet away)",
                "close (within arm's reach)",
                "very close (nearly touching)"
            ]
            for depth, category in zip(self.depth_cutoffs, depth_categories):
                if avg_depth < depth:
                    depth_category = category
                    break
            else:
                depth_category = depth_categories[-1]

            frame_height, frame_width = depth_map.shape[:2]
            horiz_pos = "left" if center_x < frame_width / 3 else "center" if center_x < 2 * frame_width / 3 else "right"
            vert_pos = "top" if center_y < frame_height / 3 else "middle" if center_y < 2 * frame_height / 3 else "bottom"
            position = f"{vert_pos} {horiz_pos}"
            is_new = self._is_new_object(label, (center_x, center_y))

            objects.append({
                "label": label,
                "score": score,
                "box": box,
                "center": (center_x, center_y),
                "depth": avg_depth,
                "depth_category": depth_category,
                "position": position,
                "size": size,
                "is_new": is_new
            })

        self._update_tracked_objects(objects)
        return objects

    def _is_new_object(self, label, center):
        if self.frame_count <= 15:
            return False
        for prev_obj in self.previous_objects:
            if prev_obj["label"] == label:
                prev_center = prev_obj["center"]
                dist = np.linalg.norm(np.subtract(center, prev_center))
                if dist < 50:
                    return False
        return True

    def _update_tracked_objects(self, current_objects):
        self.previous_objects = current_objects

    def _analyze_spatial_relationships(self, objects):
        relationships = defaultdict(list)
        if len(objects) < 2:
            return relationships
        for i, obj1 in enumerate(objects):
            for j, obj2 in enumerate(objects[i+1:], i+1):
                center1 = obj1["center"]
                center2 = obj2["center"]
                distance = np.linalg.norm(np.subtract(center1, center2))
                if distance < self.proximity_threshold * max(obj1["box"][2]-obj1["box"][0], obj2["box"][2]-obj2["box"][0]):
                    relationships["nearby"].append((obj1["label"], obj2["label"]))
                if abs(center1[1] - center2[1]) < 30:
                    if center1[0] < center2[0]:
                        relationships["side_by_side"].append((obj1["label"], "left of", obj2["label"]))
                    else:
                        relationships["side_by_side"].append((obj2["label"], "left of", obj1["label"]))
                depth_diff = abs(obj1["depth"] - obj2["depth"])
                if depth_diff > 0.2 and set(obj1["position"].split()) & set(obj2["position"].split()):
                    if obj1["depth"] > obj2["depth"]:
                        relationships["depth_order"].append((obj1["label"], "in front of", obj2["label"]))
                    else:
                        relationships["depth_order"].append((obj2["label"], "in front of", obj1["label"]))
        return relationships

    def _generate_scene_description(self, objects, relationships, caption):
        description = f"{caption}"
        if objects:
            sorted_objects = sorted(objects, key=lambda x: x["size"] * x["score"], reverse=True)
            main_objects = sorted_objects[:min(4, len(sorted_objects))]
            new_objects = [obj for obj in main_objects if obj["is_new"]]
            if new_objects:
                description += "\n\nNew in view: " + ", ".join([f"{obj['label']} ({obj['position']})" for obj in new_objects])
            if main_objects:
                primary = main_objects[0]
                description += f"\n\nPrimary object: {primary['label']} at {primary['position']}, {primary['depth_category']}."
                if len(main_objects) > 1:
                    secondary_desc = "\nAlso visible: " + ", ".join([
                        f"{obj['label']} ({obj['position']}, {obj['depth_category']})" for obj in main_objects[1:]
                    ])
                    description += secondary_desc
        if relationships:
            relation_texts = []
            for key in ["nearby", "side_by_side", "depth_order"]:
                if relationships[key]:
                    for rel in relationships[key][:2]:
                        if key == "nearby":
                            relation_texts.append(f"{rel[0]} is near {rel[1]}")
                        else:
                            relation_texts.append(f"{rel[0]} is {rel[1]} {rel[2]}")
            if relation_texts:
                description += "\n\nSpatial information: " + ". ".join(relation_texts)
        return description

def integrate_with_main(frame, detection_results, depth_map, segmentation_map, caption):
    if not hasattr(integrate_with_main, "context_builder"):
        integrate_with_main.context_builder = ContextBuilder()
    return integrate_with_main.context_builder.process_frame_data(
        detection_results, depth_map, segmentation_map, caption
    )