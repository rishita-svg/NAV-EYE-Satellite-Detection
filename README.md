# NAV-EYE V1.0 | Satellite Intelligence
An advanced ship detection system combining a Flask/TensorFlow backend with a high-end Glassmorphism UI.

### Key Technical Features:
* **Non-Maximum Suppression (NMS):** Merges overlapping detections into a single precise target lock.
* **Dynamic Bounding Boxes:** Analyzes pixel clusters to wrap rectangles tightly around ship hulls.
* **Environmental Guard:** Custom logic to filter out cloud interference and sky-line false positives.
* **Optimized Inference:** Uses batch processing to scan large satellite sectors in real-time.
