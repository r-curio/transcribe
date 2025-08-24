# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("automatic-speech-recognition", model="Khalsuu/filipino-wav2vec2-l-xls-r-300m-official")