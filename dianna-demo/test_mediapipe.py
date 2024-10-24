"""Run a performance experiment combining face detection with
facial action unit prediction for real-time processing.

Disclaimer: This code was created with the help of the generative AI tool
Claude (3.5; Sonnet) and verified for correctness.

"""

import argparse
import csv
import os
import time
from collections import deque
from datetime import datetime

import cv2
import mediapipe as mp
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.models import ResNet50_Weights, resnet50

from mexca.video.extraction import MEFARG


class FaceDetector:
    def __init__(
        self,
        process_nth_frame=1,
        model_path=None,
        confidence_threshold=0.5,
        device="cpu",
    ):
        self.device = torch.device(device)
        print(f"Using device: {self.device}")

        # Initialize MediaPipe Face Detection
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            min_detection_confidence=confidence_threshold
        )

        # Frame processing interval
        self.process_nth_frame = process_nth_frame

        # Initialize ResNet50 and move to specified device
        self.extractor = MEFARG.from_pretrained(
            "mexca/mefarg-open-graph-au-resnet50-stage-2"
        )
        self.extractor.to(self.device)
        self.extractor.eval()

        # Performance tracking
        self.fps_buffer = deque(maxlen=30)
        self.detection_times = deque(maxlen=30)
        self.inference_times = deque(maxlen=30)

        # Transform for ResNet50
        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        # CSV logging with device info
        self.csv_file = f"performance_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        with open(self.csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "Timestamp",
                    "Frame_Number",
                    "FPS",
                    "Detection_Time_ms",
                    "Inference_Time_ms",
                    "Faces_Detected",
                    "Processed",
                    "Device",
                ]
            )

        # Log system info
        self.log_system_info()

    def log_system_info(self):
        """Log system information about available GPU resources"""
        with open(self.csv_file, "a", newline="") as f:
            writer = csv.writer(f)

            # Log CUDA information if available
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    device_name = torch.cuda.get_device_name(i)
                    total_memory = (
                        torch.cuda.get_device_properties(i).total_memory
                        / 1024**3
                    )  # Convert to GB
                    writer.writerow(
                        [
                            "System Info",
                            f"GPU {i}",
                            device_name,
                            f"{total_memory:.2f} GB",
                            "",
                            "",
                            "",
                            "",
                        ]
                    )
            else:
                writer.writerow(
                    [
                        "System Info",
                        "No CUDA GPUs available",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                    ]
                )

    def process_batch_faces(self, face_tensors):
        """Process multiple faces through ResNet50 in a single batch"""
        if not face_tensors:
            return []

        # Stack all face tensors into a single batch
        batch = torch.stack(face_tensors).to(self.device)

        with torch.no_grad():
            outputs = self.extractor(batch)
            _, predictions = torch.max(outputs.data, 1)

        return predictions.cpu().numpy()

    def process_frame(self, frame, frame_number):
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width = frame.shape[:2]

        # Check if we should process this frame
        should_process = frame_number % self.process_nth_frame == 0

        faces = []
        detection_time = 0
        inference_time = 0

        if should_process:
            # Face detection timing
            detection_start = time.time()
            results = self.face_detection.process(frame_rgb)
            detection_time = (time.time() - detection_start) * 1000  # ms
            self.detection_times.append(detection_time)

            face_tensors = []

            if results.detections:
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    x = int(bbox.xmin * width)
                    y = int(bbox.ymin * height)
                    w = int(bbox.width * width)
                    h = int(bbox.height * height)

                    # Ensure coordinates are within frame boundaries
                    x = max(0, x)
                    y = max(0, y)
                    w = min(w, width - x)
                    h = min(h, height - y)

                    # Crop face
                    face = frame[y : y + h, x : x + w]
                    if face.size > 0:  # Check if face crop is valid
                        try:
                            # Transform face for ResNet
                            face_tensor = self.transform(face)
                            face_tensors.append(face_tensor)

                            face_info = {
                                "crop": face,
                                "bbox": (x, y, w, h),
                                "confidence": detection.score[0],
                            }
                            faces.append(face_info)
                        except Exception as e:
                            print(f"Error processing face: {e}")

            # Batch process faces through ResNet50
            if faces:
                inference_start = time.time()
                predictions = self.process_batch_faces(face_tensors)
                for face, prediction in zip(faces, predictions):
                    face["prediction"] = prediction
                inference_time = (time.time() - inference_start) * 1000  # ms
                self.inference_times.append(inference_time)

        return faces, detection_time, inference_time, should_process

    def process_video(self, video_path, output_path=None):
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        prev_frame_time = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Initialize video writer if output path is provided
        writer = None
        if output_path:
            os.makedirs(
                os.path.dirname(os.path.abspath(output_path)), exist_ok=True
            )
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        print(f"Processing video: {video_path}")
        print(f"Total frames: {total_frames}")
        print(f"Processing every {self.process_nth_frame} frame(s)")

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Calculate FPS
                current_time = time.time()
                fps = (
                    1 / (current_time - prev_frame_time)
                    if prev_frame_time
                    else 0
                )
                prev_frame_time = current_time
                self.fps_buffer.append(fps)

                # Process frame
                (
                    faces,
                    detection_time,
                    inference_time,
                    was_processed,
                ) = self.process_frame(frame, frame_count)

                # Write frame if output path is provided
                if writer is not None:
                    # Annotate frame if it was processed, otherwise use original frame
                    if was_processed:
                        annotated_frame = self.annotate_frame(
                            frame, faces, detection_time, inference_time, fps
                        )
                        writer.write(annotated_frame)
                    else:
                        writer.write(frame)

                # Log metrics
                self.log_metrics(
                    frame_count,
                    len(faces),
                    fps,
                    detection_time,
                    inference_time,
                    was_processed,
                )

                # Print progress
                if frame_count % 100 == 0:
                    progress = (frame_count / total_frames) * 100
                    print(
                        f"Progress: {progress:.1f}% ({frame_count}/{total_frames} frames)"
                    )

                frame_count += 1

        except KeyboardInterrupt:
            print("\nProcessing interrupted by user")
        finally:
            # Clean up
            cap.release()
            if writer is not None:
                writer.release()

            print("\nProcessing complete!")
            print(f"Performance metrics saved to: {self.csv_file}")
            if output_path:
                print(f"Annotated video saved to: {output_path}")

    def log_metrics(
        self,
        frame_number,
        num_faces,
        fps,
        detection_time,
        inference_time,
        was_processed,
    ):
        avg_fps = np.mean(self.fps_buffer) if self.fps_buffer else 0

        with open(self.csv_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
                    frame_number,
                    f"{avg_fps:.2f}",
                    f"{detection_time:.2f}",
                    f"{inference_time:.2f}",
                    num_faces,
                    was_processed,
                    str(self.device),
                ]
            )

    def annotate_frame(self, frame, faces, detection_time, inference_time, fps):
        """Add visual annotations to the frame"""
        annotated_frame = frame.copy()
        height, width = frame.shape[:2]

        # Add semi-transparent overlay for metrics
        overlay = annotated_frame.copy()
        cv2.rectangle(overlay, (0, 0), (300, 140), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.3, annotated_frame, 0.7, 0, annotated_frame)

        # Add metrics text
        cv2.putText(
            annotated_frame,
            f"Device: {self.device}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            annotated_frame,
            f"FPS: {int(fps)}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            annotated_frame,
            f"Detection: {detection_time:.1f}ms",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            annotated_frame,
            f"Inference: {inference_time:.1f}ms",
            (10, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        # Draw face boxes and predictions
        for face in faces:
            x, y, w, h = face["bbox"]
            confidence = face.get("confidence", 0)
            prediction = face.get("prediction", None)

            # Draw face rectangle
            cv2.rectangle(
                annotated_frame, (x, y), (x + w, y + h), (0, 255, 0), 2
            )

            # Draw confidence and prediction
            label = f"Conf: {confidence:.2f}"
            if prediction is not None:
                label += f" Class: {prediction}"
            cv2.putText(
                annotated_frame,
                label,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

        return annotated_frame


def main():
    parser = argparse.ArgumentParser(
        description="Process video file for face detection"
    )
    parser.add_argument(
        "video_path", type=str, help="Path to the input video file"
    )
    parser.add_argument(
        "--output", type=str, help="Path to save the annotated video (optional)"
    )
    parser.add_argument(
        "--nth-frame",
        type=int,
        default=1,
        help="Process every nth frame (default: 1)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="Face detection confidence threshold (default: 0.5)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help='Device to run on (e.g., "cpu", "cuda", "cuda:0", "cuda:1") (default: cpu)',
    )

    args = parser.parse_args()

    # Validate device
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA is not available. Falling back to CPU.")
        args.device = "cpu"

    detector = FaceDetector(
        process_nth_frame=args.nth_frame,
        confidence_threshold=args.confidence,
        device=args.device,
    )
    detector.process_video(args.video_path, args.output)


if __name__ == "__main__":
    main()
