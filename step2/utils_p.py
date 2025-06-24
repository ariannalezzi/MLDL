import torch
import numpy as np
import time
from fvcore.nn import FlopCountAnalysis, flop_count_table

def poly_lr_scheduler(optimizer, init_lr, iter, lr_decay_iter=1,
                      max_iter=300, power=0.9):
    """Polynomial decay of learning rate
            :param init_lr is base learning rate
            :param iter is a current iteration
            :param lr_decay_iter how frequently decay occurs, default is 1
            :param max_iter is number of maximum iterations
            :param power is a polymomial power

    """
    # if iter % lr_decay_iter or iter > max_iter:
    # 	return optimizer

    lr = init_lr*(1 - iter/max_iter)**power
    optimizer.param_groups[0]['lr'] = lr
    return lr
    # return lr


def fast_hist(a, b, n):
    '''
    a and b are label and prediction respectively
    n is the number of classes
    '''
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)


def per_class_iou(hist):
    epsilon = 1e-5
    return (np.diag(hist)) / (hist.sum(1) + hist.sum(0) - np.diag(hist) + epsilon)

#FLOPS and params

def analyze_flops(model, height, width):

    model = model.cpu()
    image = torch.zeros((1, 3, height, width))  # added batch dimension
    flops = FlopCountAnalysis(model, image)
    print(flop_count_table(flops))


def benchmark_model(model, height, width, iterations=1000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

    model = model.to(device).eval()
    image = torch.rand((1, 3, height, width)).to(device)

    latency = []
    fps = []

    # Warm-up (to stabilize performance)
    with torch.no_grad():
        for _ in range(10):
            _ = model(image)

    # Benchmark
    with torch.no_grad():
        for _ in range(iterations):
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.time()
            _ = model(image)
            if device.type == "cuda":
                torch.cuda.synchronize()
            end = time.time()

            elapsed = end - start
            latency.append(elapsed)
            fps.append(1.0 / elapsed if elapsed > 0 else 0.0)

    latency_ms = [l * 1000 for l in latency]
    results = {
        "mean_latency_ms": float(np.mean(latency_ms)),
        "std_latency_ms": float(np.std(latency_ms)),
        "mean_fps": float(np.mean(fps)),
        "std_fps": float(np.std(fps))
    }

    return results