# Keras
def fix_seeds(seed: int = 42) -> None:
    """
    https://keras.io/getting_started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development
    """
    import random
    import numpy as np
    import tensorflow as tf
    np.random.seed(seed)
    random.seed(seed)
    session_conf = tf.ConfigProto(
        intra_op_parallelism_threads=1,
        inter_op_parallelism_threads=1,
    )
    
    from keras import backend as K
    tf.set_random_seed(seed)
    sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    K.set_session(sess)
