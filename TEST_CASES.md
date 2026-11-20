# Milestone 4: Test Cases

| ID | Input | Condition | Expected | Actual | Pass |
| :--- | :--- | :--- | :--- | :--- | :--- |
| TC01 | Webcam | Indoor lighting, 1-2 feet away | Recognize user | Recognized user | ✓ |
| TC02 | Webcam | Person not in database | "Unknown" | "Unknown" | ✓ |
| TC03 | Webcam | Dimly lit room | "Unknown" | "Unknown" | ✓ |
| TC04 | Webcam | Half of face covered | "Unknown" | "Unknown" | ✓ |
| TC05 | Webcam | Face turned 90 degrees | "Unknown" | "Unknown" | ✓ |
| TC06 | CSV Log | After recognition | New row appended | New row appended | ✓ |
| TC07 | Duplicate | Recognize same person twice | No duplicate row | No duplicate row | ✓ |
| TC08 | UI Register | Click register, wait 3s | Photo saved | Photo saved | ✓ |
| TC09 | UI Buttons | Click Stop, then Start | Camera restarts | Camera restarts | ✓ |
| TC10 | Webcam | Stand 5 feet away | Recognize user | Distance too high to recognize | ✗ |
