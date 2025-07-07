import numpy as np

def ap_per_class(tp, conf, p_cls, t_cls, samples=1000, eps=1e-8):
    i = np.argsort(-conf)
    tp, conf, p_cls, = tp[i], conf[i], p_cls[i]

    unique_classes, nt = np.unique(t_cls, return_counts=True)
    nc = unique_classes.shape[0]

    x, prec_values = np.linspace(0, 1, samples), []

    ap, p_curve, r_curve = np.zeros((nc, tp.shape[1])), np.zeros((nc, samples)), np.zeros((nc, samples))
    for ci, c in enumerate(unique_classes):
        i = p_cls == c
        nl = nt[ci]
        np = i.sum()
        if np == 0 or nl == 0:
            continue

        fpc = (1 - tp[i]).cumsum(0)
        tpc = tp[i].cumsum(0)

        recall = tpc / (nl + eps)
        r_curve[ci] = np.interp(-x, -conf[i], recall[:, 0], left=0)

        precision = tpc / (tpc + fpc)
        p_curve[ci] = np.interp(-x, -conf[i], precision[:, 0], left=1)

        for j in range(tp.shape[1]):
            ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])
            if j == 0:
                prec_values.append(np.interp(x, mrec, mpre))
    
    prec_values = np.array(prec_values) if prec_values else np.zeros((1, 1000))

    f1_curve = 2 * p_curve * r_curve / (p_curve + r_curve + eps)

    i = smooth(f1_curve.mean(0), 0.1).argmax()
    p, r, f1 = p_curve[:, i], r_curve[:, i], f1_curve[:, i]
    return p, r, f1, ap, p_curve, r_curve, f1_curve, x, prec_values

