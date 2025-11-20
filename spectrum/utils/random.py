def set_random_state(seed: int = 42) -> None:
    import numpy as np
    import random

    random.seed(seed)
    np.random.seed(seed)
