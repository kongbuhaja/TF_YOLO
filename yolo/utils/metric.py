import numpy as np

def ap_per_class(tp, conf, p_cls, t_cls, eps=1e-8):
    i = np.argsort(-conf)
    tp, conf, p_cls, = tp[i], conf[i], p_cls[i]

    unique_classes, nt = np.unique(t_cls, return_counts=True)
    nc = unique_classes.shape[0]

    x, prec_values = np.linspace(0, 1, 1000)