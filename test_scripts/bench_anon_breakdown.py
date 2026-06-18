"""Decompose the ~47ms BLUR anonymize step into its sub-stages."""
from __future__ import annotations
import sys, time
from pathlib import Path
import cv2, numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ai_core.face_detection.face_detector import FaceDetector
from ai_core.face_parsing.face_parser import FaceParser
from ai_core.face_alignment.face_aligner import FaceAligner
from ai_core.face_anonymization.face_anonymizer import FaceAnonymizer, AnonymizationMethod, ObfuscationParams
from ai_core.face_tracking.face_tracker import ByteTracker

def now(): return time.perf_counter()
def bench(label, fn, it=60):
    fn(); fn()
    ts=[]
    for _ in range(it):
        t=now(); fn(); ts.append((now()-t)*1000)
    ts.sort(); print(f"  {label:<40} {ts[len(ts)//2]:7.2f} ms"); return ts[len(ts)//2]

cap=cv2.VideoCapture(str(ROOT/"test_videos/hai1.mp4")); ok,frame=cap.read(); cap.release()
frame=cv2.resize(frame,(640,360))
det=FaceDetector(onnx_path=str(ROOT/"ai_core/face_detection/onnx/retinaface_best.onnx"))
parser=FaceParser(); aligner=FaceAligner(output_size=(256,256),mode="ffhq")
anon=FaceAnonymizer(face_swapper=None,face_parser=parser,face_aligner=aligner)
tracks=ByteTracker().update(det.detect(frame))
p=ObfuscationParams(blur_strength=31)

print(f"faces: {len(tracks)}  (BiSeNet runs once PER face PER frame)\n")
print("=== sub-stages of one BLUR anonymize call ===")
bench("_region_mask  (align+BiSeNet parse+warp)", lambda: anon._region_mask(frame, tracks, p))
bench("GaussianBlur whole frame (31x31)", lambda: cv2.GaussianBlur(frame,(31,31),0))
blurred=cv2.GaussianBlur(frame,(31,31),0)
bench("_destroy (quantize+noise) whole frame", lambda: anon._destroy(blurred, p))
mask=anon._region_mask(frame,tracks,p)
bench("_composite float32 blend whole frame", lambda: anon._composite(frame, blurred, mask))
print()
bench(">>> total BLUR anonymize", lambda: anon.anonymize(frame,tracks,method=AnonymizationMethod.BLUR,params=p))

# Isolate just the BiSeNet model call on a 256x256 aligned crop
from ai_core.face_detection.face_detector import FaceDetection, FaceLandmarks
d0=det.detect(frame)[0]
aligned=aligner.align_detection(d0)
crop=aligner.warp_face(frame, aligned.matrix)
crop_rgb=cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
print()
bench("BiSeNet compute_mask alone (1 crop, 512x512)", lambda: parser.compute_mask(crop_rgb))

# Effect of blur_strength on GaussianBlur
print("\n=== GaussianBlur cost vs kernel size (whole frame) ===")
for k in (15,31,51,99):
    bench(f"GaussianBlur {k}x{k}", lambda k=k: cv2.GaussianBlur(frame,(k,k),0))
