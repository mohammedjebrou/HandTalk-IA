# HandTalk-IA

HandTalk-IA is a real-time sign language translator that uses a webcam, MediaPipe landmark detection, and a Transformer-based model to recognize hand gestures and speak the corresponding text.

## Features

- Real-time webcam input using OpenCV
- Hand and pose landmark extraction with MediaPipe
- Transformer-based sign gesture classification
- Text-to-speech output with gTTS
- Simple desktop UI using pygame

## Requirements

Install the Python dependencies listed in `requirement.txt`:

```bash
pip install -r requirement.txt
```

If you prefer to install manually, the required packages are:

- `opencv-python`
- `mediapipe`
- `torch`
- `gtts`
- `pygame`
- `numpy`

## Files

- `demo.py` — main application script
- `best_transformer.pt` — trained Transformer model weights
- `label_map.json` — label mapping for sign classes
- `requirement.txt` — dependency list

## Usage

1. Ensure `best_transformer.pt` and `label_map.json` are in the same folder as `demo.py`.
2. Activate your Python environment.
3. Run the application:

```bash
python demo.py
```

4. Position yourself in front of the webcam and perform sign gestures.

## Notes

- The model runs on CPU by default.
- The application expects 64-frame sign sequences and processes hand and upper-body landmarks.
- For best results, use a well-lit environment and keep your hands visible.

## Troubleshooting

- If the webcam is not detected, make sure no other application is using it.
- If MediaPipe fails to import, confirm the package is installed in the current Python environment.
- If TTS audio does not play, check that `pygame` is installed correctly.

## License

This project is provided as-is for research and experimentation.
