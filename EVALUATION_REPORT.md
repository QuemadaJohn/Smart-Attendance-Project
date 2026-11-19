# Milestone 3: Core Implementation and Testing

## 1. Testing Log (Failure Gallery)
| ID | Cause | Description | Screenshot |
| :--- | :--- | :--- | :--- |
| F01 | Lighting | Room was dark, model returned "Unknown" | `outputs/failures/dark.jpg` |
| F02 | Look-alike / Backlight | Bright light behind user caused false match to "Brad Pitt" | `outputs/failures/false_positive.jpg` |
| F03 | Viewpoint | User turned 90 degrees, model failed to detect | `outputs/failures/side.jpg` |
| F04 | Occlusion | Hand over face, model returned "Unknown" | `outputs/failures/mask.jpg` |

*(Note: The largest failure group was Look-alike/Backlight. This was fixed by lowering the cosine distance threshold to 0.35).*

## 2. Evaluation Metrics
*   **Model:** VGG-Face + RetinaFace
*   **Test Set:** 17 images (Celebrity Dataset)
*   **Speed:** ~10-15 FPS on Windows Laptop (measured after warm-up).
*   **Accuracy:** 100% (When testing the dataset against itself, all distances were below 0.35).

*(Paste the output of `evaluate.py` here)*

## 3. Improvement Plan
| Failure Cause | Change | Expected Effect |
| :--- | :--- | :--- |
| **Backlighting (False Positives)** | Add a face distance threshold `< 0.35` in `app.py`. | Prevents marking people who only partially match a celebrity. |
| **Dark Room (Missed Detections)** | Apply CLAHE (Histogram Equalization) to frames before sending to AI. | Improves recognition in low light. |
| **Viewpoint** | Register users with multiple angles (front and side). | Reduces "Unknown" errors when users move. |
