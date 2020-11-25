# IAMOnDo Dataset 
The Dataset used for training the object detection pipeline is the IAMOnDo Dataset. 
This is a dataset of online handwritten documents available as InkML files.
"inkml2img.py" is used to generate images and annotations of bounding boxes and text from InkML files.
This data will be used ot train the OCR algorithm and the Object Detection Algorithm to be used in the MobileScanner
The following classes of objects are present in the dataset:
- TextBlock
  - Textline
    - Word
    - Correction
- Drawing
- Diagram
  - Drawing
  - Textline
    - Word
    - Correction
  - Structure
  - Arrow
  - Formula
- List
  - Textline
    - Word
    - Correction
- Formula
- Textline
- Table
  - Textline
    - Word
    - Correction
- Arrow
- Structure
- Symbol
- Correction
- Word
