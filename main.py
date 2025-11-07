import streamlit as st
import cv2
import numpy as np
from pathlib import Path
import tempfile
import time
from datetime import datetime
import pandas as pd
import json
import io
import os
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
import torch
import torchvision.transforms as transforms
from scipy.optimize import linear_sum_assignment
from scipy.ndimage import gaussian_filter
from filterpy.kalman import KalmanFilter
from ultralytics import YOLO


# ==================== CORE SYSTEM: DATA STRUCTURES ====================

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


# ==================== CORE SYSTEM: HEATMAP GENERATOR ====================

class HeatmapGenerator:
    """Generate movement heatmap and trajectory visualization"""
    
    def __init__(self, frame_width: int, frame_height: int, decay_rate: float = 0.98):
        self.width = frame_width
        self.height = frame_height
        self.decay_rate = decay_rate
        self.heatmap = np.zeros((frame_height, frame_width), dtype=np.float32)
        self.trajectories = defaultdict(lambda: deque(maxlen=100))
        self.colormap = cv2.COLORMAP_JET
    
    def update(self, detections: List[Detection]):
        """Update heatmap with new detections"""
        self.heatmap *= self.decay_rate
        
        for det in detections:
            if det.track_id is None:
                continue
            
            x1, y1, x2, y2 = map(int, det.bbox)
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            self.trajectories[det.track_id].append((center_x, center_y))
            self._add_gaussian_blob(center_x, center_y, radius=30)
    
    def _add_gaussian_blob(self, x: int, y: int, radius: int = 30):
        """Add gaussian blob to heatmap"""
        y1 = max(0, y - radius)
        y2 = min(self.height, y + radius)
        x1 = max(0, x - radius)
        x2 = min(self.width, x + radius)
        
        if y2 <= y1 or x2 <= x1:
            return
        
        Y, X = np.ogrid[y1:y2, x1:x2]
        dist = np.sqrt((X - x)**2 + (Y - y)**2)
        gaussian = np.exp(-(dist**2) / (2 * (radius/3)**2))
        
        self.heatmap[y1:y2, x1:x2] += gaussian * 10
    
    def get_heatmap_overlay(self, frame: np.ndarray, alpha: float = 0.6) -> np.ndarray:
        """Get heatmap overlay on frame"""
        normalized = cv2.normalize(self.heatmap, None, 0, 255, cv2.NORM_MINMAX)
        normalized = normalized.astype(np.uint8)
        normalized = gaussian_filter(normalized, sigma=2)
        normalized = normalized.astype(np.uint8)
        heatmap_colored = cv2.applyColorMap(normalized, self.colormap)
        overlay = cv2.addWeighted(frame, 1 - alpha, heatmap_colored, alpha, 0)
        return overlay
    
    def draw_trajectories(self, frame: np.ndarray, colors: Dict[int, Tuple[int, int, int]]):
        """Draw trajectory paths"""
        for track_id, trajectory in self.trajectories.items():
            if len(trajectory) < 2:
                continue
            
            color = colors.get(track_id, (255, 255, 255))
            points = list(trajectory)
            
            for i in range(1, len(points)):
                thickness = int(np.sqrt(float(i+1)) * 1.5)
                cv2.line(frame, points[i-1], points[i], color, thickness)
            
            if points:
                cv2.circle(frame, points[-1], 5, color, -1)
                cv2.circle(frame, points[-1], 7, (255, 255, 255), 2)
    
    def reset(self):
        """Reset heatmap"""
        self.heatmap = np.zeros((self.height, self.width), dtype=np.float32)
        self.trajectories.clear()


# ==================== CORE SYSTEM: COUNTING LINE ====================

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
        cv2.line(frame, self.start_point, self.end_point, color, thickness)
        cv2.circle(frame, self.start_point, 8, color, -1)
        cv2.circle(frame, self.end_point, 8, color, -1)
        
        mid_x = (self.start_point[0] + self.end_point[0]) // 2
        mid_y = (self.start_point[1] + self.end_point[1]) // 2
        
        label = self.name
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        font_thickness = 2
        
        (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, font_thickness)
        
        cv2.rectangle(frame, 
                     (mid_x - text_width // 2 - 5, mid_y - text_height - 10),
                     (mid_x + text_width // 2 + 5, mid_y + 5),
                     color, -1)
        
        cv2.putText(frame, label, (mid_x - text_width // 2, mid_y), 
                   font, font_scale, (0, 0, 0), font_thickness)
        
        if show_direction:
            arrow_length = 50
            
            entry_x = mid_x - int(self.normal[0] * arrow_length)
            entry_y = mid_y - int(self.normal[1] * arrow_length)
            cv2.arrowedLine(frame, (entry_x, entry_y), (mid_x, mid_y),
                          (0, 255, 0), 3, tipLength=0.4)
            cv2.putText(frame, "ENTRY", (entry_x - 30, entry_y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            exit_x = mid_x + int(self.normal[0] * arrow_length)
            exit_y = mid_y + int(self.normal[1] * arrow_length)
            cv2.arrowedLine(frame, (mid_x, mid_y), (exit_x, exit_y),
                          (0, 0, 255), 3, tipLength=0.4)
            cv2.putText(frame, "EXIT", (exit_x - 25, exit_y + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)


# ==================== CORE SYSTEM: FEATURE EXTRACTION ====================

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


# ==================== CORE SYSTEM: KALMAN TRACKER ====================

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


# ==================== CORE SYSTEM: OVERLAP RESOLVER ====================

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


# ==================== CORE SYSTEM: DEEPSORT TRACKER ====================

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


# ==================== CORE SYSTEM: FOOTFALL COUNTER ====================

class FootfallCounterSystem:
    """Complete Footfall Counter with Heatmap & Trajectory Visualization"""
    
    def __init__(self, model_path='yolov8n.pt', confidence_threshold=0.5):
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.feature_extractor = FeatureExtractor()
        self.tracker = DeepSORTTracker(max_age=50, min_hits=3, appearance_threshold=0.5)
        self.overlap_resolver = OverlapResolver()
        self.counting_lines = []
        self.stats = FootfallStats()
        self.colors = {}
        self.previous_positions = {}
        self.frame_number = 0
        self.heatmap_generator = None
        self.show_heatmap = True
        self.show_trajectories = True
    
    def add_counting_line(self, line: CountingLine):
        """Add a counting line to the system"""
        self.counting_lines.append(line)
    
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
            
            x1, y1, x2, y2 = det.bbox
            curr_center = ((x1 + x2) // 2, (y1 + y2) // 2)
            
            if det.track_id in self.previous_positions:
                prev_center = self.previous_positions[det.track_id]
                
                for line in self.counting_lines:
                    crossing = line.is_crossed(prev_center, curr_center)
                    
                    if crossing:
                        if det.track_id not in self.stats.counted_ids[crossing]:
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
                            else:
                                self.stats.total_exits += 1
                                self.stats.current_count -= 1
            
            self.previous_positions[det.track_id] = curr_center
    
    def process_frame(self, frame):
        """Complete processing pipeline"""
        self.frame_number += 1
        
        if self.heatmap_generator is None:
            h, w = frame.shape[:2]
            self.heatmap_generator = HeatmapGenerator(w, h)
        
        detections = self.detect_humans(frame)
        detections = self.overlap_resolver.resolve_overlaps(detections)
        detections = self.tracker.update(detections)
        self.heatmap_generator.update(detections)
        self.check_line_crossings(detections)
        
        return detections
    
    def visualize(self, frame, detections):
        """Draw everything on frame"""
        if self.show_heatmap:
            frame = self.heatmap_generator.get_heatmap_overlay(frame, alpha=0.5)
        
        if self.show_trajectories:
            self.heatmap_generator.draw_trajectories(frame, self.colors)
        
        for line in self.counting_lines:
            line.draw(frame, show_direction=True)
        
        for det in detections:
            if det.track_id is None:
                continue
            
            if det.track_id not in self.colors:
                self.colors[det.track_id] = tuple(np.random.randint(50, 255, 3).tolist())
            color = self.colors[det.track_id]
            
            x1, y1, x2, y2 = map(int, det.bbox)
            
            crossed_entry = det.track_id in self.stats.counted_ids['entry']
            crossed_exit = det.track_id in self.stats.counted_ids['exit']
            
            if crossed_entry and not crossed_exit:
                box_color = (0, 255, 0)
                status = "ENTERED"
            elif crossed_exit:
                box_color = (0, 0, 255)
                status = "EXITED"
            else:
                box_color = color
                status = "TRACKING"
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 3)
            
            label = f"ID:{det.track_id} | {status}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            
            cv2.rectangle(frame, (x1, y1-label_size[1]-10), 
                         (x1+label_size[0]+10, y1), box_color, -1)
            cv2.putText(frame, label, (x1+5, y1-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        self.draw_stats_panel(frame)
        
        return frame
    
    def draw_stats_panel(self, frame):
        """Draw statistics panel on frame"""
        h, w = frame.shape[:2]
        
        overlay = frame.copy()
        panel_height = 220
        cv2.rectangle(overlay, (0, 0), (w, panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
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


# ==================== STREAMLIT UI ====================

# Page configuration
st.set_page_config(
    page_title="AI Footfall Counter Pro",
    page_icon="🚶‍♂️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional Custom CSS
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    
    .main-title {
        font-size: 3rem;
        font-weight: 700;
        color: white;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-subtitle {
        font-size: 1.2rem;
        color: #e0e0e0;
        margin-top: 0.5rem;
    }
    
    .stats-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
        transition: transform 0.3s ease;
    }
    
    .stats-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    
    .metric-value {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .metric-label {
        font-size: 1rem;
        color: #666;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .info-box {
        background: linear-gradient(135deg, #e0f7fa 0%, #b2ebf2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #00acc1;
        margin: 1rem 0;
    }
    
    .success-box {
        background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #4caf50;
        margin: 1rem 0;
    }
    
    .warning-box {
        background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #ff9800;
        margin: 1rem 0;
    }
    
    .sidebar-section {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border: 2px solid #e0e0e0;
    }
    
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        font-size: 1rem;
        font-weight: 600;
        border-radius: 8px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    .footer {
        text-align: center;
        padding: 2rem;
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 10px;
        margin-top: 2rem;
    }
    
    .video-container {
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    </style>
""", unsafe_allow_html=True)


# Initialize session state
if 'system' not in st.session_state:
    st.session_state.system = None
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'video_loaded' not in st.session_state:
    st.session_state.video_loaded = False
if 'show_heatmap' not in st.session_state:
    st.session_state.show_heatmap = True
if 'show_trajectories' not in st.session_state:
    st.session_state.show_trajectories = True
if 'stop_requested' not in st.session_state:
    st.session_state.stop_requested = False


# Video library path - UPDATE THIS PATH
VIDEO_LIBRARY_PATH = "videos"  # Change this to your video folder path


def get_available_videos():
    """Get list of available videos from the video library"""
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv']
    
    if not os.path.exists(VIDEO_LIBRARY_PATH):
        os.makedirs(VIDEO_LIBRARY_PATH, exist_ok=True)
        return []
    
    videos = []
    for file in os.listdir(VIDEO_LIBRARY_PATH):
        if any(file.lower().endswith(ext) for ext in video_extensions):
            videos.append(file)
    
    return sorted(videos)


def initialize_system(confidence):
    """Initialize the footfall counter system"""
    if st.session_state.system is None:
        with st.spinner("🚀 Initializing AI Footfall Counter System..."):
            st.session_state.system = FootfallCounterSystem(
                model_path='yolov8n.pt',
                confidence_threshold=confidence
            )
            st.success("✅ System initialized successfully!")


def setup_counting_line(frame_width, frame_height):
    """Setup horizontal center counting line"""
    line = CountingLine(
        start_point=(50, frame_height // 2),
        end_point=(frame_width - 50, frame_height // 2),
        name="Center Line"
    )
    st.session_state.system.add_counting_line(line)


def process_webcam():
    """Process webcam stream"""
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        st.error("❌ Error: Could not open webcam. Please check your camera connection.")
        return
    
    ret, first_frame = cap.read()
    if not ret:
        st.error("❌ Error: Could not read from webcam")
        cap.release()
        return
    
    frame_height, frame_width = first_frame.shape[:2]
    setup_counting_line(frame_width, frame_height)
    
    video_placeholder = st.empty()
    stats_placeholder = st.empty()
    
    st.session_state.processing = True
    frame_count = 0
    fps_time = time.time()
    
    while st.session_state.processing and not st.session_state.stop_requested:
        ret, frame = cap.read()
        if not ret:
            break
        
        st.session_state.system.show_heatmap = st.session_state.show_heatmap
        st.session_state.system.show_trajectories = st.session_state.show_trajectories
        
        detections = st.session_state.system.process_frame(frame)
        frame = st.session_state.system.visualize(frame, detections)
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
        
        frame_count += 1
        if frame_count % 30 == 0:
            fps = 30 / (time.time() - fps_time)
            fps_time = time.time()
            
            stats_placeholder.markdown(f"""
                <div class="info-box">
                    <h4>📊 Real-time Processing</h4>
                    <p><strong>FPS:</strong> {fps:.1f} | <strong>Frames:</strong> {frame_count}</p>
                </div>
            """, unsafe_allow_html=True)
    
    cap.release()
    st.session_state.processing = False


def process_video(video_path):
    """Process video file"""
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        st.error("❌ Error: Could not open video file")
        return
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    
    ret, first_frame = cap.read()
    if not ret:
        st.error("❌ Error: Could not read video file")
        cap.release()
        return
    
    frame_height, frame_width = first_frame.shape[:2]
    setup_counting_line(frame_width, frame_height)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    video_placeholder = st.empty()
    progress_bar = st.progress(0)
    stats_placeholder = st.empty()
    
    st.session_state.processing = True
    frame_count = 0
    fps_time = time.time()
    
    while st.session_state.processing and not st.session_state.stop_requested:
        ret, frame = cap.read()
        if not ret:
            break
        
        st.session_state.system.show_heatmap = st.session_state.show_heatmap
        st.session_state.system.show_trajectories = st.session_state.show_trajectories
        
        detections = st.session_state.system.process_frame(frame)
        frame = st.session_state.system.visualize(frame, detections)
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
        
        frame_count += 1
        progress = frame_count / total_frames
        progress_bar.progress(progress)
        
        if frame_count % 30 == 0:
            current_fps = 30 / (time.time() - fps_time)
            fps_time = time.time()
            
            stats_placeholder.markdown(f"""
                <div class="info-box">
                    <h4>📊 Processing Video</h4>
                    <p><strong>Frame:</strong> {frame_count}/{total_frames} | 
                    <strong>FPS:</strong> {current_fps:.1f} | 
                    <strong>Progress:</strong> {progress*100:.1f}%</p>
                </div>
            """, unsafe_allow_html=True)
    
    cap.release()
    progress_bar.progress(1.0)
    st.session_state.processing = False


def display_final_stats():
    """Display final statistics"""
    if st.session_state.system is None:
        return
    
    stats = st.session_state.system.stats
    
    st.markdown("---")
    st.markdown("## 📊 Final Statistics Dashboard")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
            <div class="stats-card">
                <div class="metric-value" style="color: #4caf50;">{stats.total_entries}</div>
                <div class="metric-label">Total Entries</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
            <div class="stats-card">
                <div class="metric-value" style="color: #f44336;">{stats.total_exits}</div>
                <div class="metric-label">Total Exits</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
            <div class="stats-card">
                <div class="metric-value" style="color: #ff9800;">{stats.current_count}</div>
                <div class="metric-label">Current Count</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
            <div class="stats-card">
                <div class="metric-value" style="color: #2196f3;">{len(st.session_state.system.colors)}</div>
                <div class="metric-label">Unique Persons</div>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("### 📋 Detailed Analytics")
    info_col1, info_col2 = st.columns(2)
    
    with info_col1:
        st.markdown(f"""
            <div class="info-box">
                <h4>⚙️ Processing Details</h4>
                <ul style="list-style: none; padding-left: 0;">
                    <li>📹 <strong>Total Frames:</strong> {st.session_state.system.frame_number}</li>
                    <li>📝 <strong>Total Events:</strong> {len(stats.events)}</li>
                    <li>🔥 <strong>Heatmap:</strong> {'Enabled ✅' if st.session_state.show_heatmap else 'Disabled ❌'}</li>
                    <li>📍 <strong>Trajectories:</strong> {'Enabled ✅' if st.session_state.show_trajectories else 'Disabled ❌'}</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    
    with info_col2:
        st.markdown(f"""
            <div class="success-box">
                <h4>✅ Counting Summary</h4>
                <ul style="list-style: none; padding-left: 0;">
                    <li>🟢 <strong>Entry IDs:</strong> {len(stats.counted_ids['entry'])}</li>
                    <li>🔴 <strong>Exit IDs:</strong> {len(stats.counted_ids['exit'])}</li>
                    <li>📊 <strong>Net Count:</strong> {stats.total_entries - stats.total_exits}</li>
                    <li>⏰ <strong>Timestamp:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)


def create_csv_download():
    """Create CSV file for download"""
    if st.session_state.system is None or not st.session_state.system.stats.events:
        return None
    
    events_data = []
    for event in st.session_state.system.stats.events:
        events_data.append({
            'Track_ID': event.track_id,
            'Timestamp': event.timestamp,
            'Direction': event.direction,
            'Line_Name': event.line_name,
            'Frame_Number': event.frame_number
        })
    
    df = pd.DataFrame(events_data)
    return df.to_csv(index=False).encode('utf-8')


def create_json_download():
    """Create JSON file for download"""
    if st.session_state.system is None:
        return None
    
    stats = st.session_state.system.stats
    
    summary = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'total_entries': stats.total_entries,
        'total_exits': stats.total_exits,
        'current_count': stats.current_count,
        'total_unique_persons': len(st.session_state.system.colors),
        'total_frames_processed': st.session_state.system.frame_number,
        'counting_lines': [
            {
                'name': line.name,
                'start': line.start_point,
                'end': line.end_point
            }
            for line in st.session_state.system.counting_lines
        ],
        'events_count': len(stats.events),
        'events': [
            {
                'track_id': event.track_id,
                'timestamp': event.timestamp,
                'direction': event.direction,
                'line_name': event.line_name,
                'frame_number': event.frame_number
            }
            for event in stats.events
        ]
    }
    
    return json.dumps(summary, indent=2).encode('utf-8')


def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown("""
        <div class="main-header">
            <h1 class="main-title">🚶‍♂️ AI Footfall Counter Pro</h1>
            <p class="main-subtitle">Advanced People Counting with Real-time Heatmap & Trajectory Visualization</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("# ⚙️ Control Panel")
        
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown("### 📹 Input Source")
        
        input_source = st.radio(
            "Select your input:",
            ["📁 Video Library", "📤 Upload Video", "📷 Webcam"],
            index=0,
            help="Choose where to get your video from"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        video_path = None
        
        if input_source == "📁 Video Library":
            st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
            available_videos = get_available_videos()
            
            if available_videos:
                selected_video = st.selectbox(
                    "Select a video:",
                    available_videos,
                    help=f"Videos from: {VIDEO_LIBRARY_PATH}"
                )
                video_path = os.path.join(VIDEO_LIBRARY_PATH, selected_video)
                st.session_state.video_loaded = True
                st.success(f"✅ Selected: {selected_video}")
            else:
                st.warning(f"⚠️ No videos found in `{VIDEO_LIBRARY_PATH}` folder. Please add some videos!")
                st.info("💡 Supported formats: MP4, AVI, MOV, MKV, WMV, FLV")
            st.markdown('</div>', unsafe_allow_html=True)
        
        elif input_source == "📤 Upload Video":
            st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
            uploaded_file = st.file_uploader(
                "Upload your video file:",
                type=['mp4', 'avi', 'mov', 'mkv', 'wmv', 'flv'],
                help="Upload a video file to process"
            )
            
            if uploaded_file is not None:
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                tfile.write(uploaded_file.read())
                video_path = tfile.name
                st.session_state.video_loaded = True
                st.success(f"✅ Uploaded: {uploaded_file.name}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown("### 🎨 Visualization")
        
        st.session_state.show_heatmap = st.checkbox(
            "🔥 Heatmap Overlay",
            value=st.session_state.show_heatmap,
            help="Show movement intensity heatmap"
        )
        
        st.session_state.show_trajectories = st.checkbox(
            "📍 Track Trajectories",
            value=st.session_state.show_trajectories,
            help="Display person movement paths"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown("### 🤖 AI Model Settings")
        
        confidence = st.slider(
            "Detection Confidence:",
            min_value=0.1,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Minimum confidence threshold for person detection"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        if st.button("🚀 Initialize System", type="primary", use_container_width=True):
            initialize_system(confidence)
            if st.session_state.system is not None:
                st.session_state.system.confidence_threshold = confidence
    
    # Main content
    col_main, col_stats = st.columns([3, 1])
    
    with col_main:
        st.markdown("## 🎬 Video Processing Center")
        
        btn_col1, btn_col2, btn_col3 = st.columns(3)
        
        with btn_col1:
            start_disabled = st.session_state.processing or st.session_state.system is None
            if input_source != "📷 Webcam":
                start_disabled = start_disabled or video_path is None
            
            start_button = st.button(
                "▶️ START PROCESSING",
                type="primary",
                disabled=start_disabled,
                use_container_width=True
            )
        
        with btn_col2:
            stop_button = st.button(
                "⏹️ STOP",
                type="secondary",
                disabled=not st.session_state.processing,
                use_container_width=True
            )
        
        with btn_col3:
            reset_button = st.button(
                "🔄 RESET SYSTEM",
                disabled=st.session_state.processing,
                use_container_width=True
            )
        
        if start_button:
            if st.session_state.system is None:
                st.error("❌ Please initialize the system first from the sidebar!")
            elif input_source != "📷 Webcam" and video_path is None:
                st.error("❌ Please select or upload a video file first!")
            else:
                st.session_state.stop_requested = False
                
                if input_source == "📷 Webcam":
                    st.info("📷 Starting webcam... Please allow camera access if prompted.")
                    process_webcam()
                else:
                    process_video(video_path)
                
                if not st.session_state.stop_requested:
                    st.success("✅ Processing completed successfully!")
                else:
                    st.warning("⚠️ Processing stopped by user")
        
        if stop_button:
            st.session_state.stop_requested = True
            st.session_state.processing = False
            st.warning("⏹️ Stopping processing...")
        
        if reset_button:
            st.session_state.system = None
            st.session_state.processing = False
            st.session_state.video_loaded = False
            st.session_state.stop_requested = False
            st.success("🔄 System reset complete!")
            st.rerun()
    
    with col_stats:
        st.markdown("## 📊 Live Stats")
        
        if st.session_state.system is not None:
            stats = st.session_state.system.stats
            
            st.markdown(f"""
                <div class="stats-card">
                    <div class="metric-value" style="color: #4caf50; font-size: 2rem;">{stats.total_entries}</div>
                    <div class="metric-label" style="font-size: 0.8rem;">Entries</div>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
                <div class="stats-card">
                    <div class="metric-value" style="color: #f44336; font-size: 2rem;">{stats.total_exits}</div>
                    <div class="metric-label" style="font-size: 0.8rem;">Exits</div>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
                <div class="stats-card">
                    <div class="metric-value" style="color: #ff9800; font-size: 2rem;">{stats.current_count}</div>
                    <div class="metric-label" style="font-size: 0.8rem;">Current</div>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
                <div class="stats-card">
                    <div class="metric-value" style="color: #2196f3; font-size: 2rem;">{len(st.session_state.system.colors)}</div>
                    <div class="metric-label" style="font-size: 0.8rem;">Tracked</div>
                </div>
            """, unsafe_allow_html=True)
            
            if st.session_state.system.frame_number > 0:
                st.markdown(f"""
                    <div class="stats-card">
                        <div class="metric-value" style="color: #9c27b0; font-size: 2rem;">{st.session_state.system.frame_number}</div>
                        <div class="metric-label" style="font-size: 0.8rem;">Frames</div>
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.info("💡 Initialize the system to view live statistics")
    
    # Display final statistics
    if st.session_state.system is not None and not st.session_state.processing and st.session_state.system.frame_number > 0:
        display_final_stats()
        
        # Download section
        st.markdown("---")
        st.markdown("## 💾 Export Analytics Data")
        
        col_download1, col_download2 = st.columns(2)
        
        with col_download1:
            csv_data = create_csv_download()
            if csv_data:
                st.download_button(
                    label="📥 Download Events (CSV)",
                    data=csv_data,
                    file_name=f"footfall_events_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True,
                    help="Download detailed event log in CSV format"
                )
        
        with col_download2:
            json_data = create_json_download()
            if json_data:
                st.download_button(
                    label="📥 Download Summary (JSON)",
                    data=json_data,
                    file_name=f"footfall_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True,
                    help="Download complete summary in JSON format"
                )
        
        # Event log table
        if st.session_state.system.stats.events:
            st.markdown("### 📋 Event Log")
            
            with st.expander("View Recent Events", expanded=False):
                events_data = []
                for event in st.session_state.system.stats.events[-50:]:  # Show last 50 events
                    events_data.append({
                        'Track ID': event.track_id,
                        'Direction': '🟢 Entry' if event.direction == 'entry' else '🔴 Exit',
                        'Time': event.timestamp,
                        'Line': event.line_name,
                        'Frame': event.frame_number
                    })
                
                df = pd.DataFrame(events_data)
                st.dataframe(df, use_container_width=True, hide_index=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div class="footer">
            <h3>🚀 AI Footfall Counter Pro v2.0</h3>
            <p style="color: #666; margin-top: 0.5rem;">
                Powered by YOLOv8 Deep Learning & DeepSORT Multi-Object Tracking
            </p>
            <p style="color: #888; margin-top: 0.5rem; font-size: 0.9rem;">
                🔬 Advanced Computer Vision • 📊 Real-time Analytics • 🎯 High Accuracy Tracking
            </p>
        </div>
    """, unsafe_allow_html=True)
    
   
if __name__ == "__main__":
    main()