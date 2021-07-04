import pickle
import matplotlib.pyplot as plt
import numpy as np


def plot_regret(plotdir, log):
    with open('ccs/MinecartDeterministic-v0.pkl', 'rb') as f:
        ccs = pickle.load(f)
    gamma = 1.0

    r = np.stack([log[f'train/reward/{i}'][:,1] for i in range(3)], axis=-1)
    w_step = log['train/weights/step'][:len(r)]
    w = log['train/weights/ndarray'][:len(r)]
    limit = np.abs(w_step-1000000).argmin()
    r =r[:limit]
    w_step = w_step[:limit]
    w = w[:limit]

    us = np.sum(w*ccs[gamma][None], -1)
    max_u = np.max(us, axis=1)

    u = np.sum(w[:,0]*r, -1)

    regret = max_u-u
    cum_regret = np.cumsum(regret)

    plt.figure()
    plt.plot(w_step, regret)
    plt.savefig(plotdir / 'ep_regret.png')
    plt.close()
    plt.figure()
    plt.plot(w_step, cum_regret)
    plt.savefig(plotdir / 'ep_cum_regret.png')
    plt.close()
