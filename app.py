import cv2
import numpy as np
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
import time
import torch
import torchvision.transforms as transforms
from datetime import datetime
import pandas as pd
import json
from scipy.optimize import linear_sum_assignment
from scipy.ndimage import gaussian_filter
from filterpy.kalman import KalmanFilter
from ultralytics import YOLO


# ==================== DATA STRUCTURES ====================

@dataclass
class Detection:
    """Detection data structure"""
    bbox: np.ndarray
    confidence: float
    class_id: int
    track_id: Optional[int] = None
    feature: Optional[np.ndarray] = None


@dataclass
class CountingEvent:
    """Store counting event details"""
    track_id: int
    timestamp: str
    direction: str  # 'entry' or 'exit'
    line_name: str
    frame_number: int


@dataclass
class FootfallStats:
    """Footfall statistics"""
    total_entries: int = 0
    total_exits: int = 0
    current_count: int = 0
    events: List[CountingEvent] = field(default_factory=list)
    counted_ids: Dict[str, set] = field(default_factory=lambda: {'entry': set(), 'exit': set()})


# ==================== HEATMAP GENERATOR ====================

class HeatmapGenerator:
    """Generate movement heatmap and trajectory visualization"""
    
    def __init__(self, frame_width: int, frame_height: int, decay_rate: float = 0.98):
        self.width = frame_width
        self.height = frame_height
        self.decay_rate = decay_rate
        
        # Heatmap matrix
        self.heatmap = np.zeros((frame_height, frame_width), dtype=np.float32)
        
        # Trajectory storage
        self.trajectories = defaultdict(lambda: deque(maxlen=100))
        
        # Color map for visualization
        self.colormap = cv2.COLORMAP_JET
    
    def update(self, detections: List[Detection]):
        """Update heatmap with new detections"""
        # Apply decay
        self.heatmap *= self.decay_rate
        
        # Add new detections
        for det in detections:
            if det.track_id is None:
                continue
            
            x1, y1, x2, y2 = map(int, det.bbox)
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            # Update trajectory
            self.trajectories[det.track_id].append((center_x, center_y))
            
            # Add gaussian blob to heatmap
            self._add_gaussian_blob(center_x, center_y, radius=30)
    
    def _add_gaussian_blob(self, x: int, y: int, radius: int = 30):
        """Add gaussian blob to heatmap"""
        y1 = max(0, y - radius)
        y2 = min(self.height, y + radius)
        x1 = max(0, x - radius)
        x2 = min(self.width, x + radius)
        
        if y2 <= y1 or x2 <= x1:
            return
        
        # Create gaussian kernel
        Y, X = np.ogrid[y1:y2, x1:x2]
        dist = np.sqrt((X - x)**2 + (Y - y)**2)
        gaussian = np.exp(-(dist**2) / (2 * (radius/3)**2))
        
        # Add to heatmap
        self.heatmap[y1:y2, x1:x2] += gaussian * 10
    
    def get_heatmap_overlay(self, frame: np.ndarray, alpha: float = 0.6) -> np.ndarray:
        """Get heatmap overlay on frame"""
        # Normalize heatmap
        normalized = cv2.normalize(self.heatmap, None, 0, 255, cv2.NORM_MINMAX)
        normalized = normalized.astype(np.uint8)
        
        # Apply gaussian blur for smoothness
        normalized = gaussian_filter(normalized, sigma=2)
        normalized = normalized.astype(np.uint8)
        
        # Apply colormap
        heatmap_colored = cv2.applyColorMap(normalized, self.colormap)
        
        # Blend with original frame
        overlay = cv2.addWeighted(frame, 1 - alpha, heatmap_colored, alpha, 0)
        
        return overlay
    
    def draw_trajectories(self, frame: np.ndarray, colors: Dict[int, Tuple[int, int, int]]):
        """Draw trajectory paths"""
        for track_id, trajectory in self.trajectories.items():
            if len(trajectory) < 2:
                continue
            
            color = colors.get(track_id, (255, 255, 255))
            points = list(trajectory)
            
            # Draw trajectory line
            for i in range(1, len(points)):
                # Thickness decreases for older points
                thickness = int(np.sqrt(float(i+1)) * 1.5)
                alpha = i / len(points)  # Fade older points
                
                # Draw line segment
                cv2.line(frame, points[i-1], points[i], color, thickness)
            
            # Draw current position marker
            if points:
                cv2.circle(frame, points[-1], 5, color, -1)
                cv2.circle(frame, points[-1], 7, (255, 255, 255), 2)
    
    def reset(self):
        """Reset heatmap"""
        self.heatmap = np.zeros((self.height, self.width), dtype=np.float32)
        self.trajectories.clear()


# ==================== COUNTING LINE ====================

class CountingLine:
    """Virtual counting line for footfall detection"""
    
    def __init__(self, start_point: Tuple[int, int], end_point: Tuple[int, int], 
                 name: str = "Counting Line"):
        self.start_point = start_point
        self.end_point = end_point
        self.name = name
        self._calculate_line_properties()
    
    def _calculate_line_properties(self):
        """Calculate line direction vector and normal"""
        x1, y1 = self.start_point
        x2, y2 = self.end_point
        
        self.direction = np.array([x2 - x1, y2 - y1], dtype=float)
        length = np.linalg.norm(self.direction)
        
        if length > 0:
            self.direction = self.direction / length
        
        self.normal = np.array([-self.direction[1], self.direction[0]])
    
    def get_side(self, point: Tuple[int, int]) -> float:
        """Determine which side of the line a point is on"""
        x1, y1 = self.start_point
        x2, y2 = self.end_point
        px, py = point
        
        cross = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
        return cross
    
    def is_crossed(self, prev_center: Tuple[int, int], 
                   curr_center: Tuple[int, int]) -> Optional[str]:
        """Check if trajectory crosses the line and determine direction"""
        prev_side = self.get_side(prev_center)
        curr_side = self.get_side(curr_center)
        
        if prev_side * curr_side < 0:
            if prev_side < 0 and curr_side > 0:
                return 'entry'
            elif prev_side > 0 and curr_side < 0:
                return 'exit'
        
        return None
    
    def draw(self, frame: np.ndarray, color: Tuple[int, int, int] = (0, 255, 255), 
             thickness: int = 3, show_direction: bool = True):
        """Draw the counting line on frame"""
        # Draw main line
        cv2.line(frame, self.start_point, self.end_point, color, thickness)
        
        # Draw end circles
        cv2.circle(frame, self.start_point, 8, color, -1)
        cv2.circle(frame, self.end_point, 8, color, -1)
        
        # Draw label
        mid_x = (self.start_point[0] + self.end_point[0]) // 2
        mid_y = (self.start_point[1] + self.end_point[1]) // 2
        
        label = self.name
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        font_thickness = 2
        
        (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, font_thickness)
        
        # Draw text background
        cv2.rectangle(frame, 
                     (mid_x - text_width // 2 - 5, mid_y - text_height - 10),
                     (mid_x + text_width // 2 + 5, mid_y + 5),
                     color, -1)
        
        cv2.putText(frame, label, (mid_x - text_width // 2, mid_y), 
                   font, font_scale, (0, 0, 0), font_thickness)
        
        # Draw direction arrows
        if show_direction:
            arrow_length = 50
            
            # Entry arrow (green)
            entry_x = mid_x - int(self.normal[0] * arrow_length)
            entry_y = mid_y - int(self.normal[1] * arrow_length)
            cv2.arrowedLine(frame, (entry_x, entry_y), (mid_x, mid_y),
                          (0, 255, 0), 3, tipLength=0.4)
            cv2.putText(frame, "ENTRY", (entry_x - 30, entry_y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Exit arrow (red)
            exit_x = mid_x + int(self.normal[0] * arrow_length)
            exit_y = mid_y + int(self.normal[1] * arrow_length)
            cv2.arrowedLine(frame, (mid_x, mid_y), (exit_x, exit_y),
                          (0, 0, 255), 3, tipLength=0.4)
            cv2.putText(frame, "EXIT", (exit_x - 25, exit_y + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)


# ==================== FEATURE EXTRACTION ====================

class FeatureExtractor:
    """Extract appearance features for person re-identification"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        from torchvision.models import resnet18, ResNet18_Weights
        self.model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
        self.model.to(self.device)
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def extract_feature(self, frame, bbox):
        try:
            x1, y1, x2, y2 = map(int, bbox)
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            if x2 <= x1 or y2 <= y1:
                return None
            
            person_img = frame[y1:y2, x1:x2]
            if person_img.size == 0:
                return None
            
            img_tensor = self.transform(person_img).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                feature = self.model(img_tensor)
                feature = feature.squeeze().cpu().numpy()
                feature = feature / (np.linalg.norm(feature) + 1e-6)
            
            return feature
        except:
            return None
    
    @staticmethod
    def compute_similarity(feat1, feat2):
        if feat1 is None or feat2 is None:
            return 0.0
        return np.dot(feat1, feat2)


# ==================== KALMAN TRACKER ====================

class KalmanBoxTracker:
    """Kalman Filter for tracking bounding boxes"""
    count = 0
    
    def __init__(self, bbox, feature=None):
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([
            [1,0,0,0,1,0,0], [0,1,0,0,0,1,0], [0,0,1,0,0,0,1],
            [0,0,0,1,0,0,0], [0,0,0,0,1,0,0], [0,0,0,0,0,1,0],
            [0,0,0,0,0,0,1]
        ])
        self.kf.H = np.array([
            [1,0,0,0,0,0,0], [0,1,0,0,0,0,0],
            [0,0,1,0,0,0,0], [0,0,0,1,0,0,0]
        ])
        
        self.kf.R[2:,2:] *= 10.
        self.kf.P[4:,4:] *= 1000.
        self.kf.P *= 10.
        self.kf.Q[-1,-1] *= 0.01
        self.kf.Q[4:,4:] *= 0.01
        
        self.kf.x[:4] = self._convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        
        self.features = deque(maxlen=30)
        self.feature_average = feature
        if feature is not None:
            self.features.append(feature)
        
    def update(self, bbox, feature=None):
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(self._convert_bbox_to_z(bbox))
        
        if feature is not None:
            self.features.append(feature)
            if self.feature_average is None:
                self.feature_average = feature
            else:
                self.feature_average = 0.9 * self.feature_average + 0.1 * feature
                self.feature_average = self.feature_average / (np.linalg.norm(self.feature_average) + 1e-6)
        
    def predict(self):
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(self._convert_x_to_bbox(self.kf.x))
        return self.history[-1]
    
    def get_state(self):
        return self._convert_x_to_bbox(self.kf.x)
    
    def get_feature(self):
        return self.feature_average
    
    @staticmethod
    def _convert_bbox_to_z(bbox):
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = bbox[0] + w/2.
        y = bbox[1] + h/2.
        s = w * h
        r = w / float(h)
        return np.array([x, y, s, r]).reshape((4, 1))
    
    @staticmethod
    def _convert_x_to_bbox(x, score=None):
        w = np.sqrt(x[2] * x[3])
        h = x[2] / w
        if score is None:
            return np.array([x[0]-w/2., x[1]-h/2., x[0]+w/2., x[1]+h/2.]).reshape((1,4))
        else:
            return np.array([x[0]-w/2., x[1]-h/2., x[0]+w/2., x[1]+h/2., score]).reshape((1,5))


# ==================== OVERLAP RESOLVER ====================

class OverlapResolver:
    """ML-based overlap resolution"""
    
    @staticmethod
    def calculate_iou(box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    @staticmethod
    def resolve_overlaps(detections: List[Detection], iou_threshold=0.3):
        if len(detections) <= 1:
            return detections
        
        sorted_dets = sorted(detections, key=lambda x: x.confidence, reverse=True)
        keep = []
        
        for det1 in sorted_dets:
            should_keep = True
            
            for det2 in keep:
                iou = OverlapResolver.calculate_iou(det1.bbox, det2.bbox)
                
                if iou > iou_threshold:
                    conf_ratio = det1.confidence / det2.confidence
                    if 0.7 < conf_ratio < 1.3:
                        center1 = [(det1.bbox[0] + det1.bbox[2])/2, 
                                 (det1.bbox[1] + det1.bbox[3])/2]
                        center2 = [(det2.bbox[0] + det2.bbox[2])/2, 
                                 (det2.bbox[1] + det2.bbox[3])/2]
                        distance = np.sqrt((center1[0]-center2[0])**2 + 
                                         (center1[1]-center2[1])**2)
                        box_size = np.sqrt((det1.bbox[2]-det1.bbox[0])**2 + 
                                          (det1.bbox[3]-det1.bbox[1])**2)
                        
                        if distance > box_size * 0.3:
                            continue
                    
                    should_keep = False
                    break
            
            if should_keep:
                keep.append(det1)
        
        return keep


# ==================== DEEPSORT TRACKER ====================

class DeepSORTTracker:
    """Enhanced DeepSORT tracker"""
    
    def __init__(self, max_age=50, min_hits=3, iou_threshold=0.3, appearance_threshold=0.5):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.appearance_threshold = appearance_threshold
        self.trackers = []
        self.frame_count = 0
        self.lost_trackers = []
        self.max_lost_age = 100
        
    def update(self, detections: List[Detection]):
        self.frame_count += 1
        
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)
        
        if len(detections) > 0:
            dets = np.array([d.bbox for d in detections])
            features = [d.feature for d in detections]
            
            matched, unmatched_dets, unmatched_trks = self._associate_detections_to_trackers(
                dets, trks, features, self.iou_threshold
            )
            
            for m in matched:
                self.trackers[m[1]].update(dets[m[0]], features[m[0]])
                detections[m[0]].track_id = self.trackers[m[1]].id
            
            if len(unmatched_dets) > 0 and len(self.lost_trackers) > 0:
                reidentified = self._reidentify_lost_tracks(
                    [detections[i] for i in unmatched_dets]
                )
                
                for det_idx, tracker in reidentified:
                    det = detections[unmatched_dets[det_idx]]
                    tracker.update(det.bbox, det.feature)
                    tracker.time_since_update = 0
                    self.trackers.append(tracker)
                    det.track_id = tracker.id
                    self.lost_trackers.remove(tracker)
                    unmatched_dets = [i for i in unmatched_dets if i != unmatched_dets[det_idx]]
            
            for i in unmatched_dets:
                trk = KalmanBoxTracker(dets[i], features[i])
                self.trackers.append(trk)
                detections[i].track_id = trk.id
        
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            i -= 1
            if trk.time_since_update > self.max_age:
                if trk.hits >= self.min_hits:
                    self.lost_trackers.append(trk)
                self.trackers.pop(i)
        
        self.lost_trackers = [t for t in self.lost_trackers 
                             if t.time_since_update < self.max_lost_age]
        
        return detections
    
    def _reidentify_lost_tracks(self, unmatched_detections):
        reidentified = []
        
        for det_idx, det in enumerate(unmatched_detections):
            if det.feature is None:
                continue
            
            best_similarity = self.appearance_threshold
            best_tracker = None
            
            for tracker in self.lost_trackers:
                tracker_feature = tracker.get_feature()
                if tracker_feature is None:
                    continue
                
                similarity = FeatureExtractor.compute_similarity(det.feature, tracker_feature)
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_tracker = tracker
            
            if best_tracker is not None:
                reidentified.append((det_idx, best_tracker))
        
        return reidentified
    
    def _associate_detections_to_trackers(self, detections, trackers, features, iou_threshold=0.3):
        if len(trackers) == 0:
            return np.empty((0,2), dtype=int), np.arange(len(detections)), np.empty((0,5), dtype=int)
        
        iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)
        for d, det in enumerate(detections):
            for t, trk in enumerate(trackers):
                iou_matrix[d, t] = OverlapResolver.calculate_iou(det, trk)
        
        appearance_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)
        for d, det_feature in enumerate(features):
            if det_feature is not None:
                for t, tracker in enumerate(self.trackers):
                    tracker_feature = tracker.get_feature()
                    if tracker_feature is not None:
                        appearance_matrix[d, t] = FeatureExtractor.compute_similarity(
                            det_feature, tracker_feature
                        )
        
        cost_matrix = -(0.4 * iou_matrix + 0.6 * appearance_matrix)
        
        if min(cost_matrix.shape) > 0:
            matched_indices = self._linear_assignment(cost_matrix)
        else:
            matched_indices = np.empty(shape=(0,2))
        
        unmatched_detections = []
        for d in range(len(detections)):
            if d not in matched_indices[:,0]:
                unmatched_detections.append(d)
        
        unmatched_trackers = []
        for t in range(len(trackers)):
            if t not in matched_indices[:,1]:
                unmatched_trackers.append(t)
        
        matches = []
        for m in matched_indices:
            iou_ok = iou_matrix[m[0], m[1]] >= iou_threshold
            appearance_ok = appearance_matrix[m[0], m[1]] >= self.appearance_threshold
            
            if iou_ok or appearance_ok:
                matches.append(m.reshape(1,2))
            else:
                unmatched_detections.append(m[0])
                unmatched_trackers.append(m[1])
        
        if len(matches) == 0:
            matches = np.empty((0,2), dtype=int)
        else:
            matches = np.concatenate(matches, axis=0)
        
        return matches, np.array(unmatched_detections), np.array(unmatched_trackers)
    
    @staticmethod
    def _linear_assignment(cost_matrix):
        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))


# ==================== FOOTFALL COUNTER SYSTEM ====================

class FootfallCounterSystem:
    """Complete Footfall Counter with Heatmap & Trajectory Visualization"""
    
    def __init__(self, model_path='yolov8n.pt', confidence_threshold=0.5):
        print("🚀 Initializing Footfall Counter System...")
        print("=" * 70)
        
        print("📦 Loading YOLOv8 model...")
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        
        print("🧠 Loading feature extractor for re-identification...")
        self.feature_extractor = FeatureExtractor()
        
        print("🎯 Initializing tracker...")
        self.tracker = DeepSORTTracker(max_age=50, min_hits=3, appearance_threshold=0.5)
        self.overlap_resolver = OverlapResolver()
        
        # Counting lines
        self.counting_lines = []
        
        # Statistics
        self.stats = FootfallStats()
        
        # Visualization
        self.colors = {}
        self.previous_positions = {}
        
        # Frame counter
        self.frame_number = 0
        
        # Heatmap generator (initialized later with frame dimensions)
        self.heatmap_generator = None
        
        # Visualization modes
        self.show_heatmap = True
        self.show_trajectories = True
        
        print("✅ System initialized successfully!")
        print("=" * 70 + "\n")
    
    def add_counting_line(self, line: CountingLine):
        """Add a counting line to the system"""
        self.counting_lines.append(line)
        print(f"✓ Added counting line: {line.name}")
    
    def detect_humans(self, frame):
        """Detect humans and extract features"""
        results = self.model(frame, verbose=False)
        
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                if int(box.cls[0]) == 0 and float(box.conf[0]) > self.confidence_threshold:
                    bbox = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0])
                    feature = self.feature_extractor.extract_feature(frame, bbox)
                    
                    detection = Detection(
                        bbox=bbox,
                        confidence=confidence,
                        class_id=0,
                        feature=feature
                    )
                    detections.append(detection)
        
        return detections
    
    def check_line_crossings(self, detections):
        """Check if any person crossed counting lines"""
        for det in detections:
            if det.track_id is None:
                continue
            
            # Get current center
            x1, y1, x2, y2 = det.bbox
            curr_center = ((x1 + x2) // 2, (y1 + y2) // 2)
            
            # Check if we have previous position
            if det.track_id in self.previous_positions:
                prev_center = self.previous_positions[det.track_id]
                
                # Check each counting line
                for line in self.counting_lines:
                    crossing = line.is_crossed(prev_center, curr_center)
                    
                    if crossing:
                        # Check if this ID already counted for this direction
                        line_key = f"{line.name}_{crossing}"
                        
                        if det.track_id not in self.stats.counted_ids[crossing]:
                            # Record the crossing event
                            event = CountingEvent(
                                track_id=det.track_id,
                                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                direction=crossing,
                                line_name=line.name,
                                frame_number=self.frame_number
                            )
                            
                            self.stats.events.append(event)
                            self.stats.counted_ids[crossing].add(det.track_id)
                            
                            if crossing == 'entry':
                                self.stats.total_entries += 1
                                self.stats.current_count += 1
                                print(f"✓ ENTRY detected! ID:{det.track_id} | Line: {line.name} | Total Entries: {self.stats.total_entries}")
                            else:
                                self.stats.total_exits += 1
                                self.stats.current_count -= 1
                                print(f"✓ EXIT detected! ID:{det.track_id} | Line: {line.name} | Total Exits: {self.stats.total_exits}")
            
            # Update previous position
            self.previous_positions[det.track_id] = curr_center
    
    def process_frame(self, frame):
        """Complete processing pipeline"""
        self.frame_number += 1
        
        # Initialize heatmap generator if needed
        if self.heatmap_generator is None:
            h, w = frame.shape[:2]
            self.heatmap_generator = HeatmapGenerator(w, h)
        
        # Detect humans with features
        detections = self.detect_humans(frame)
        
        # Resolve overlapping detections
        detections = self.overlap_resolver.resolve_overlaps(detections)
        
        # Update tracker
        detections = self.tracker.update(detections)
        
        # Update heatmap
        self.heatmap_generator.update(detections)
        
        # Check line crossings
        self.check_line_crossings(detections)
        
        return detections
    
    def visualize(self, frame, detections):
        """Draw everything on frame"""
        # Apply heatmap overlay if enabled
        if self.show_heatmap:
            frame = self.heatmap_generator.get_heatmap_overlay(frame, alpha=0.5)
        
        # Draw trajectories if enabled
        if self.show_trajectories:
            self.heatmap_generator.draw_trajectories(frame, self.colors)
        
        # Draw counting lines
        for line in self.counting_lines:
            line.draw(frame, show_direction=True)
        
        # Draw detections
        for det in detections:
            if det.track_id is None:
                continue
            
            # Get consistent color
            if det.track_id not in self.colors:
                self.colors[det.track_id] = tuple(np.random.randint(50, 255, 3).tolist())
            color = self.colors[det.track_id]
            
            # Draw bounding box
            x1, y1, x2, y2 = map(int, det.bbox)
            
            # Check if person crossed line (highlight differently)
            crossed_entry = det.track_id in self.stats.counted_ids['entry']
            crossed_exit = det.track_id in self.stats.counted_ids['exit']
            
            if crossed_entry and not crossed_exit:
                box_color = (0, 255, 0)  # Green for entered
                status = "ENTERED"
            elif crossed_exit:
                box_color = (0, 0, 255)  # Red for exited
                status = "EXITED"
            else:
                box_color = color
                status = "TRACKING"
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 3)
            
            # Draw label
            label = f"ID:{det.track_id} | {status}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            
            cv2.rectangle(frame, (x1, y1-label_size[1]-10), 
                         (x1+label_size[0]+10, y1), box_color, -1)
            cv2.putText(frame, label, (x1+5, y1-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw statistics panel
        self.draw_stats_panel(frame)
        
        return frame
    
    def draw_stats_panel(self, frame):
        """Draw statistics panel on frame"""
        h, w = frame.shape[:2]
        
        # Create semi-transparent overlay for stats
        overlay = frame.copy()
        panel_height = 220
        cv2.rectangle(overlay, (0, 0), (w, panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Draw stats
        font = cv2.FONT_HERSHEY_SIMPLEX
        y_offset = 35
        
        stats_info = [
            f"ENTRIES: {self.stats.total_entries}",
            f"EXITS: {self.stats.total_exits}",
            f"CURRENT: {self.stats.current_count}",
            f"TRACKED IDs: {len(self.colors)}",
            f"FRAME: {self.frame_number}",
            f"HEATMAP: {'ON' if self.show_heatmap else 'OFF'} | TRAJECTORY: {'ON' if self.show_trajectories else 'OFF'}"
        ]
        
        colors_list = [(0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (255, 255, 255), (0, 255, 255)]
        
        for i, (info, color) in enumerate(zip(stats_info, colors_list)):
            cv2.putText(frame, info, (20, y_offset + i*30), 
                       font, 0.8, color, 2, cv2.LINE_AA)
    
    def export_data(self, filename_prefix="footfall_data"):
        """Export counting data to CSV and JSON"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Export events to CSV
        if self.stats.events:
            events_data = []
            for event in self.stats.events:
                events_data.append({
                    'Track_ID': event.track_id,
                    'Timestamp': event.timestamp,
                    'Direction': event.direction,
                    'Line_Name': event.line_name,
                    'Frame_Number': event.frame_number
                })
            
            df = pd.DataFrame(events_data)
            csv_filename = f"{filename_prefix}_events_{timestamp}.csv"
            df.to_csv(csv_filename, index=False)
            print(f"✓ Events exported to: {csv_filename}")
        
        # Export summary to JSON
        summary = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'total_entries': self.stats.total_entries,
            'total_exits': self.stats.total_exits,
            'current_count': self.stats.current_count,
            'total_unique_persons': len(self.colors),
            'total_frames_processed': self.frame_number,
            'counting_lines': [
                {'name': line.name, 'start': line.start_point, 'end': line.end_point}
                for line in self.counting_lines
            ],
            'events_count': len(self.stats.events)
        }
        
        json_filename = f"{filename_prefix}_summary_{timestamp}.json"
        with open(json_filename, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"✓ Summary exported to: {json_filename}")
        
        return csv_filename, json_filename
    
    def print_summary(self):
        """Print final summary"""
        print("\n" + "="*70)
        print("📊 FOOTFALL COUNTER - FINAL SUMMARY")
        print("="*70)
        print(f"Total Entries:        {self.stats.total_entries}")
        print(f"Total Exits:          {self.stats.total_exits}")
        print(f"Current Count:        {self.stats.current_count}")
        print(f"Unique Persons:       {len(self.colors)}")
        print(f"Total Events:         {len(self.stats.events)}")
        print(f"Frames Processed:     {self.frame_number}")
        print("="*70 + "\n")


# ==================== MAIN APPLICATION ====================

def main():
    """Main application"""
    print("\n" + "="*70)
    print("🎯 FOOTFALL COUNTER SYSTEM - Enhanced with Heatmap & Trajectories")
    print("="*70 + "\n")
    
    # Initialize system
    system = FootfallCounterSystem(
        model_path='yolov8n.pt',
        confidence_threshold=0.5
    )
    
    # Choose input source
    print("Select input source:")
    print("1. Webcam")
    print("2. Video file (provide path)")
    choice = input("Enter choice (1-2): ").strip()
    
    if choice == '1':
        source = 0
        print("✓ Using webcam")
    elif choice == '2':
        source = input("Enter video file path: ").strip()
        print(f"✓ Using video: {source}")
    else:
        print("Invalid choice, using webcam")
        source = 0
    
    # Open video
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("❌ Error: Could not open video source")
        return
    
    # Get first frame for line configuration
    ret, first_frame = cap.read()
    if not ret:
        print("❌ Error: Could not read frame")
        return
    
    frame_height, frame_width = first_frame.shape[:2]
    
    # Configure counting line (horizontal center only)
    print("\n" + "="*70)
    print("📏 Setting up horizontal center counting line...")
    print("="*70)
    
    # Create horizontal center line
    line = CountingLine(
        start_point=(50, frame_height // 2),
        end_point=(frame_width - 50, frame_height // 2),
        name="Center Line"
    )
    system.add_counting_line(line)
    
    # Reset video to start
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    # Process video
    print("\n" + "="*70)
    print("🎬 Processing video...")
    print("Controls:")
    print("  • Press 'q' to quit")
    print("  • Press 'p' to pause/resume")
    print("  • Press 'h' to toggle heatmap")
    print("  • Press 't' to toggle trajectories")
    print("="*70 + "\n")
    
    fps_time = time.time()
    frame_count = 0
    paused = False
    
    # Optional: Setup video writer
    save_output = input("Save output video? (y/n): ").strip().lower() == 'y'
    out = None
    
    if save_output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        out = cv2.VideoWriter('footfall_output_heatmap.mp4', fourcc, fps, 
                             (frame_width, frame_height))
        print("✓ Output will be saved to: footfall_output_heatmap.mp4")
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("\n✓ Video processing completed")
                break
            
            # Process frame
            detections = system.process_frame(frame)
            
            # Visualize
            frame = system.visualize(frame, detections)
            
            # Save frame if recording
            if out is not None:
                out.write(frame)
            
            # Calculate FPS
            frame_count += 1
            if frame_count % 30 == 0:
                fps = 30 / (time.time() - fps_time)
                fps_time = time.time()
                print(f"FPS: {fps:.1f} | Entries: {system.stats.total_entries} | "
                      f"Exits: {system.stats.total_exits} | Current: {system.stats.current_count}")
        
        # Display
        cv2.imshow('Footfall Counter - Heatmap & Trajectories', frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print("\n⚠ User stopped processing")
            break
        elif key == ord('p'):
            paused = not paused
            status = "PAUSED" if paused else "RESUMED"
            print(f"▶ {status}")
        elif key == ord('h'):
            system.show_heatmap = not system.show_heatmap
            status = "ON" if system.show_heatmap else "OFF"
            print(f"🔥 Heatmap: {status}")
        elif key == ord('t'):
            system.show_trajectories = not system.show_trajectories
            status = "ON" if system.show_trajectories else "OFF"
            print(f"📍 Trajectories: {status}")
    
    # Cleanup
    cap.release()
    if out is not None:
        out.release()
    cv2.destroyAllWindows()
    
    # Print summary
    system.print_summary()
    
    # Export data
    export = input("Export data to CSV/JSON? (y/n): ").strip().lower()
    if export == 'y':
        system.export_data()
        print("✅ Data exported successfully!")
    
    print("\n" + "="*70)
    print("✅ FOOTFALL COUNTER SYSTEM - SESSION COMPLETED")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()