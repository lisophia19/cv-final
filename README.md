# cv-final
Fridge food detection for recipe generation.

## Instructions to run code
1. Create and activate a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
2. Install dependencies:
   ```bash
   python -m pip install -U pip
   python -m pip install -U ultralytics
   ```
3. Run the script:
   ```bash
   python yolo.py
   ```

## Notes
- If you get an import warning in the editor, select `.venv/bin/python` as your Python interpreter.
- Update `path/to/dataset.yaml` in `yolo.py` to your actual dataset config before training.
