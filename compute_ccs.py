import numpy as np
import scipy
from scipy.spatial import ConvexHull
import gym
from gym.wrappers import TimeLimit
from math import ceil
import itertools
from gym.envs.multi_objective.minecart.minecart import \
    Mine, Cart, BASE_RADIUS, BASE_SCALE, MINE_RADIUS, MINE_SCALE, HOME_POS, ROTATION, ACCELERATION, \
    ACT_ACCEL, ACT_BRAKE, ACT_LEFT, ACT_RIGHT, ACT_MINE, ACT_NONE, FUEL_ACC, FUEL_IDLE, FUEL_MINE, mag


FUEL_LIST = [FUEL_MINE + FUEL_IDLE, FUEL_IDLE, FUEL_IDLE,
             FUEL_IDLE + FUEL_ACC, FUEL_IDLE, FUEL_IDLE]


def compute_angle(p0, p1, p2):
    v0 = np.array(p0) - np.array(p1)
    v1 = np.array(p2) - np.array(p1)

    angle = np.math.atan2(np.linalg.det([v0, v1]), np.dot(v0, v1))
    return np.degrees(angle)


def pareto_filter(costs, minimize=True):
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :param return_mask: True to return a mask
    :return: An array of indices of pareto-efficient points.
        If return_mask is True, this will be an (n_points, ) boolean array
        Otherwise it will be a (n_efficient_points, ) integer array of indices.
    from https://stackoverflow.com/a/40239615
    """
    costs_copy = np.copy(costs) if minimize else -np.copy(costs)
    is_efficient = np.arange(costs_copy.shape[0])
    n_points = costs_copy.shape[0]
    next_point_index = 0  # Next index in the is_efficient array to search for
    while next_point_index < len(costs_copy):
        nondominated_point_mask = np.any(
            costs_copy < costs_copy[next_point_index], axis=1)
        nondominated_point_mask[next_point_index] = True
        # Remove dominated points
        is_efficient = is_efficient[nondominated_point_mask]
        costs_copy = costs_copy[nondominated_point_mask]
        next_point_index = np.sum(
            nondominated_point_mask[:next_point_index]) + 1
    return [costs[i] for i in is_efficient]


def truncated_mean(mean, std, a, b):
    if std == 0:
        return mean
    from scipy.stats import norm
    a = (a - mean) / std
    b = (b - mean) / std
    PHIB = norm.cdf(b)
    PHIA = norm.cdf(a)
    phib = norm.pdf(b)
    phia = norm.pdf(a)

    trunc_mean = (mean + ((phia - phib) / (PHIB - PHIA)) * std)
    return trunc_mean


def convex_coverage_set(policies):
    """
        Computes an approximate convex coverage set
        Keyword Arguments:
            frame_skip {int} -- How many times each action is repeated (default: {1})
            discount {float} -- Discount factor to apply to rewards (default: {1})
            incremental_frame_skip {bool} -- Wether actions are repeated incrementally (default: {1})
            symmetric {bool} -- If true, we assume the pattern of accelerations from the base to the mine is the same as from the mine to the base (default: {True})
        Returns:
            The convex coverage set 
    """
    origin = np.min(policies, axis=0)
    extended_policies = [origin] + policies
    return [policies[idx - 1] for idx in ConvexHull(extended_policies).vertices if idx != 0]


def pareto_coverage_set(self, frame_skip=1, discount=0.98, incremental_frame_skip=True, symmetric=True):
    """
        Computes an approximate pareto coverage set
        Keyword Arguments:
            frame_skip {int} -- How many times each action is repeated (default: {1})
            discount {float} -- Discount factor to apply to rewards (default: {1})
            incremental_frame_skip {bool} -- Wether actions are repeated incrementally (default: {1})
            symmetric {bool} -- If true, we assume the pattern of accelerations from the base to the mine is the same as from the mine to the base (default: {True})
        Returns:
            The pareto coverage set
    """
    all_rewards = []
    base_perimeter = BASE_RADIUS * BASE_SCALE

    # Empty mine just outside the base
    virtual_mine = Mine(self.ore_cnt, (base_perimeter**2 / 2)
                        ** (1 / 2), (base_perimeter**2 / 2)**(1 / 2))
    virtual_mine.distributions = [
        scipy.stats.norm(0, 0)
        for _ in range(self.ore_cnt)
    ]
    for mine in (self.mines + [virtual_mine]):
        mine_distance = mag(mine.pos - HOME_POS) - \
            MINE_RADIUS * MINE_SCALE - BASE_RADIUS * BASE_SCALE / 2

        # Number of rotations required to face the mine
        angle = compute_angle(mine.pos, HOME_POS, [1, 1])
        rotations = int(ceil(abs(angle) / (ROTATION * frame_skip)))

        # Build pattern of accelerations/nops to reach the mine
        # initialize with single acceleration
        queue = [{"speed": ACCELERATION * frame_skip, "dist": mine_distance - frame_skip *
                    (frame_skip + 1) / 2 * ACCELERATION if incremental_frame_skip else mine_distance - ACCELERATION * frame_skip * frame_skip, "seq": [ACT_ACCEL]}]
        trimmed_sequences = []

        while len(queue) > 0:
            seq = queue.pop()
            # accelerate
            new_speed = seq["speed"] + ACCELERATION * frame_skip
            accelerations = new_speed / ACCELERATION
            movement = accelerations * (accelerations + 1) / 2 * ACCELERATION - (
                accelerations - frame_skip) * ((accelerations - frame_skip) + 1) / 2 * ACCELERATION
            dist = seq["dist"] - movement
            speed = new_speed
            if dist <= 0:
                trimmed_sequences.append(seq["seq"] + [ACT_ACCEL])
            else:
                queue.append({"speed": speed, "dist": dist,
                                "seq": seq["seq"] + [ACT_ACCEL]})
            # idle
            dist = seq["dist"] - seq["speed"] * frame_skip

            if dist <= 0:
                trimmed_sequences.append(seq["seq"] + [ACT_NONE])
            else:
                queue.append(
                    {"speed": seq["speed"], "dist": dist, "seq": seq["seq"] + [ACT_NONE]})

        # Build rational mining sequences
        mine_means = mine.distribution_means() * frame_skip
        mn_sum = np.sum(mine_means)
        # on average it takes up to this many actions to fill cart
        max_mine_actions = 0 if mn_sum == 0 else int(
            ceil(self.capacity / mn_sum))

        # all possible mining sequences (i.e. how many times we mine)
        mine_sequences = [[ACT_MINE] *
                            i for i in range(1, max_mine_actions + 1)]

        # All possible combinations of actions before, during and after mining
        if len(mine_sequences) > 0:
            if not symmetric:
                all_sequences = map(
                    lambda sequences: list(sequences[0]) + list(sequences[1]) + list(
                        sequences[2]) + list(
                        sequences[3]) + list(
                        sequences[4]),
                    itertools.product([[ACT_LEFT] * rotations],
                                        trimmed_sequences,
                                        [[ACT_BRAKE] + [ACT_LEFT] *
                                            (180 // (ROTATION * frame_skip))],
                                        mine_sequences,
                                        trimmed_sequences)
                )

            else:
                all_sequences = map(
                    lambda sequences: list(sequences[0]) + list(sequences[1]) + list(
                        sequences[2]) + list(
                        sequences[3]) + list(
                        sequences[1]),
                    itertools.product([[ACT_LEFT] * rotations],
                                        trimmed_sequences,
                                        [[ACT_BRAKE] + [ACT_LEFT] *
                                            (180 // (ROTATION * frame_skip))],
                                        mine_sequences)
                )
        else:
            if not symmetric:
                print([ACT_NONE] + trimmed_sequences[1:],
                        trimmed_sequences[1:], trimmed_sequences)
                all_sequences = map(
                    lambda sequences: list(sequences[0]) + list(sequences[1]) + list(
                        sequences[2]) + [ACT_NONE] + list(
                        sequences[3])[1:],
                    itertools.product([[ACT_LEFT] * rotations],
                                        trimmed_sequences,
                                        [[ACT_LEFT] *
                                            (180 // (ROTATION * frame_skip))],
                                        trimmed_sequences)
                )

            else:
                all_sequences = map(
                    lambda sequences: list(sequences[0]) + list(sequences[1]) + list(
                        sequences[2]) + [ACT_NONE] + list(
                        sequences[1][1:]),
                    itertools.product([[ACT_LEFT] * rotations],
                                        trimmed_sequences,
                                        [[ACT_LEFT] * (180 // (ROTATION * frame_skip))])
                )

        # Compute rewards for each sequence
        fuel_costs = np.array([f * frame_skip for f in FUEL_LIST])

        def maxlen(l):
            if len(l) == 0:
                return 0
            return max([len(s) for s in l])

        longest_pattern = maxlen(trimmed_sequences)
        max_len = rotations + longest_pattern + 1 + \
            (180 // (ROTATION * frame_skip)) + \
            maxlen(mine_sequences) + longest_pattern
        discount_map = discount**np.arange(max_len)
        for s in all_sequences:
            reward = np.zeros((len(s), self.obj_cnt()))
            reward[:, -1] = fuel_costs[s]
            mine_actions = s.count(ACT_MINE)
            reward[-1, :-1] = mine_means * mine_actions / \
                max(1, (mn_sum * mine_actions) / self.capacity)

            reward = np.dot(discount_map[:len(s)], reward)
            all_rewards.append(reward)

        all_rewards = pareto_filter(all_rewards, minimize=False)

    return all_rewards


def minecart_ccs(gamma):
    env = gym.make('MinecartDeterministic-v0')
    policies = pareto_coverage_set(env, 4, gamma)
    ccs = convex_coverage_set(policies)
    return ccs


def dst_ccs(gamma):
    returns = np.array([
            [18, -2],
            [26, -3],
            [31, -4],
            [44, -7],
            [48.2, -8],
            [56, -10],
            [72, -14],
            [76.3, -15],
            [90, -18],
            [100, -20]
        ])
    ccs = returns[:, 0]*gamma**(-returns[:,1]-1), [np.sum(-gamma**np.arange(-i)) for i in returns[:,1]]
    ccs = np.stack(ccs, 1)
    # ccs = convex_coverage_set(ccs)
    return ccs

if __name__ == '__main__':
    import pickle

    # ccs_fun = {
    #     'dst': dst_ccs,
    #     'minecart': minecart_ccs
    # }
    # for env in ['dst', 'minecart']:
    #     all_ccs = {}
    #     for gamma in [0.95, 0.98, 0.99, 1.]:
    #         ccs = ccs_fun[env](gamma)
    #         ccs = np.array(ccs)
    #         all_ccs[gamma] = ccs
    #     with open(f'{env}.pkl', 'wb') as f:
    #         pickle.dump(all_ccs, f)

    #     print(all_ccs)

    f = open('minecart.pkl', 'rb')
    inf = pickle.load(f)
    print(inf)