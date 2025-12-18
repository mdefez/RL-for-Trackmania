keys = ["W", "A", "S", "D"]

import random

class Agent :
    def __init__(self):
        self.history = []

    def feed(self, data) -> list[str]:
        self.history.append(data)
        return random.sample(keys, k=random.randint(0, len(keys)))

