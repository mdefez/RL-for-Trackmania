import pandas as pd

blocks = pd.read_csv("ParseTMMap/blocks.csv", sep = ";", encoding="utf-16")

blocks["center_x"] = (blocks["X"] + 0.5) * 32.0
blocks["center_y"] = (blocks["Y"] + 0.5) * 8.0
blocks["center_z"] = (blocks["Z"] + 0.5) * 32.0


