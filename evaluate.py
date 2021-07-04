import os
import numpy as np
import pandas as pd
import pickle

import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator


OPT={
    "r1": [0.85192097, 0.923623597, 0.65784273, 0.47126046, 0.55392683, 0.57676679, 0.45001093, 0.16992796, 0.18423008, 0],
    "r2": [0.16992796, 0.18423008, 0.45001093, 0.47126046, 0.55392683, 0.57676679, 0.65784273, 0.85192097, 0.923623597, 0],
    "r3": [-0.81650404, -0.94917702, -1.11527075, -0.70880496, -0.79616165, -1.11842114, -1.11527075, -0.81650404, -0.94917702, -0.24923698]
}

OPT_1={
    "r1": [ 0.55612687,  0.85192097,  0.60293373,  0.92362357,  0.40115733,  0.58274497,
   0.63179215,  0.65784273,  0.47126046,  0.55392683,  0.57676679,  0.27441997,
   0.39863875,  0.43219049,  0.45001093,  0.11092755,  0.16992796,  0.12026385,
   0.18423008,  0.        ],
    "r2": [ 0.11092755,  0.16992796,  0.12026385,  0.18423008,  0.27441997,  0.39863875,
   0.43219049,  0.45001093,  0.47126046,  0.55392683,  0.57676679,  0.40115733,
   0.58274497,  0.63179215,  0.65784273,  0.55612687,  0.85192097,  0.60293373,
   0.92362357,  0.        ],
    "r3": [-0.65061296, -0.81650404, -0.77743027, -0.94917702, -0.67291286, -0.83521908,
  -0.96483263, -1.11527075, -0.70880496, -0.79616165, -1.11842114, -0.67291286,
  -0.83521908, -0.96483263, -1.11527075, -0.65061296, -0.81650404, -0.77743027,
  -0.94917702, -0.24923698]
  }

OPT_R = np.array([OPT['r1'], OPT['r2'], OPT['r3']]).T

def parse_array(text):
    array = text.lstrip(' [').rstrip(' ]').split()
    array = [eval(a) for a in array]
    return array

def episode_evaluate(file_path):
    regret_list = []
    steps_list = []
    weight_list = []
    opt_scal_reward_list = []
    act_scal_reward_list = []
    error_list = []
    act_reward_list = []
    opt_reward_list = []
    error = 0
    total_eps = 0
    total_reward = 0
    total_regret = 0

    with open(file_path, 'r') as fin:
        for line in fin.readlines():
            line = line.rstrip('\n')
            log = line.split(';')
            if log[0] == 'episode':
                steps_list.append(log[1])
                weight = parse_array(log[-2])
                weight_list.append(weight)
                error = log[-4]

                act_scal_reward = eval(log[5])
                act_scal_reward_list.append(act_scal_reward)

                act_reward = parse_array(log[7])
                act_reward_list.append(act_reward)

                opt_scal_reward = max(np.dot(OPT_R, weight))
                opt_scal_reward_list.append(opt_scal_reward)

                opt_reward = OPT_R[np.argmax(np.dot(OPT_R, weight))]
                opt_reward_list.append(opt_reward)

                total_eps += 1

                total_reward += act_scal_reward
                # if opt_reward - scal_reward < 0:
                #     error += (opt_reward - scal_reward)
                total_regret += (opt_scal_reward - act_scal_reward)
                regret_list.append(total_regret)
                error_list.append(error)


    df = pd.DataFrame({'step':steps_list, 'regret':regret_list, 'weight':weight_list, 'act_scal_reward':act_scal_reward_list, 'act_reward':act_reward_list, 'opt_scal_reward':opt_scal_reward_list, 'opt_reward':opt_reward_list, 'error':error_list})
    df.to_csv(file_path+'.csv')

def logs_evaluate(file_path):
    log_file = file_path
    steps_list = []

    error_list = []

    with open(log_file, 'r') as fin:
        for line in fin.readlines():
            line = line.rstrip('\n')
            log = line.split(';')
            if log[0] == 'logs':
                steps_list.append(log[1])
                weight = parse_array(log[-2])
                error = log[-5]

                error_list.append(error)

    df = pd.DataFrame({'step':steps_list, 'error':error_list})
    df.to_csv(log_file+'.csv')

def cal_adhesion(file_path):
    steps_list = list()
    adhesion_list = list()

    with open(file_path, 'r') as fin:
        for line in fin.readlines():
            line = line.rstrip('\n')
            log = line.split(';')
            batch_size = int(log[1])
            steps_list.append(log[0])
            adhesion = 0
            for i in eval(log[2]):
                adhesion += np.linalg.norm(np.array(i[0])-np.array(parse_array(log[-1])))*i[1]
            adhesion_list.append(adhesion/batch_size)

            # if log[0] == stop:
            #     break


    plt.plot(steps_list[::8], adhesion_list[::8], color='navy', linewidth=3, alpha=0.4)

    mean_adhesion_list = [sum(adhesion_list[i:i+8])/8 for i in range(0,len(adhesion_list)-8,8)]
    plt.plot(steps_list[:-8:8], mean_adhesion_list, color='indigo')

    plt.title('adhesion degree')
    plt.xlabel('steps')
    plt.ylabel('adhesion degree')

    x_major_locator = MultipleLocator(160)
    ax = plt.gca()

    plt.tick_params(axis='x', labelsize=6)
    plt.xticks(rotation=30)
    ax.xaxis.set_major_locator(x_major_locator)
    plt.xlim(0, 3200)

    for i in range(0, 10, 1):
        plt.hlines(i/10, 0, len(steps_list), colors = "black", linestyles = "dashed")
    # plt.savefig(log_file+'.jpg')

    # plt.legend()
    plt.show()
    plt.close()

def cal_adhesion_2(file_path, record_range, sample_interval=1):
    steps_list = list()
    adhesion_list = list()

    with open(file_path, 'r') as fin:
        for line in fin.readlines():
            line = line.rstrip('\n')
            log = line.split(';')
            
            if int(log[0]) >= record_range[0]:
                batch_size = int(log[1])
                steps_list.append(log[0])
                adhesion = 0
                for i in eval(log[2]):
                    adhesion += np.linalg.norm(np.array(i[0])-np.array(parse_array(log[-1]))) * i[1]
                adhesion_list.append(adhesion/batch_size)
                
            if int(log[0]) >= record_range[1]:
                break

    plt.plot(steps_list[::sample_interval], adhesion_list[::sample_interval])
    plt.title('adhesion degree')
    plt.xlabel('steps')
    plt.ylabel('adhesion degree')

    x_major_locator = MultipleLocator((record_range[1]-record_range[0])/100)
    ax = plt.gca()

    plt.tick_params(axis='x', labelsize=6)
    plt.xticks(rotation=30)
    ax.xaxis.set_major_locator(x_major_locator)

    for i in range(0, 10, 1):
        plt.hlines(i/10, 0, len(steps_list), colors = "black", linestyles = "dashed")
    # plt.tick_params(axis='x', labelsize=6)
    # plt.xticks(rotation=30)
    # plt.xlim(record_range[0], record_range[1])

    plt.show()
    plt.close()

def draw_episodes(file_path):
    data = pd.read_csv(file_path+'.csv')

    step_list = data['step'].to_list()
    new_steps_list = []
    regret_list = []
    error_list = []
    act_treasure = []
    opt_treasure = []

    flag = [i for i in range(step_list[0], step_list[-1], 100)]

    new_steps_list.append(step_list[0])
    regret_list.append(data['regret'][0])
    error_list.append(data['error'][0])
    act_treasure.append(data['act_reward'][0])
    opt_treasure.append(data['opt_reward'][0])
    j = 1

    for i in range(1, len(step_list)-1, 1):

        if j == len(flag):
            break

        while step_list[i] - flag[j] >= 99:
            j = j + 1
        
        if step_list[i] >= (flag[j] - 99) and step_list[i] <= (flag[j] + 99):
            if abs(step_list[i+1] - flag[j]) < abs(step_list[i] - flag[j]):
                continue
            else:
                new_steps_list.append(step_list[i])
                regret_list.append(data['regret'][i])
                error_list.append(data['error'][i])
                act_treasure.append(data['act_reward'][i])
                opt_treasure.append(data['opt_reward'][i])
                j = j + 1

    # regret
    ax1 = plt.subplot(1, 2, 1)
    plt.sca(ax1)
    plt.plot(new_steps_list, regret_list)

    plt.title('Total Regret')
    plt.xlabel('steps')
    plt.ylabel('regret')

    x_major_locator = MultipleLocator(100000)
    ax = plt.gca()

    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

    ax.spines['bottom'].set_position(('data',0))  

    plt.tick_params(axis='x', labelsize=6)
    ax.xaxis.set_major_locator(x_major_locator)
    plt.xlim(0, 1000000)
    for i in range(0, 1000001, 50000):
        plt.vlines(i, 0, 10000, colors = "black", linestyles = "dashed")


    # error
    ax2 = plt.subplot(1, 2, 2)
    plt.sca(ax2)
    plt.plot(new_steps_list, error_list)
    plt.title('error')
    plt.xlabel('steps')
    plt.ylabel('error')

    x_major_locator = MultipleLocator(100000)
    ax = plt.gca()

    ax.spines['bottom'].set_position(('data',0))  

    plt.tick_params(axis='x', labelsize=6)
    ax.xaxis.set_major_locator(x_major_locator)
    plt.xlim(0, 1000000)
    
    # plt.savefig(log_file+'.jpg')
    plt.show()

# logs_file_path = os.path.join(os.getcwd(), 'output/logs/rewards_AP_2-regular')
# transitions_file_path = os.path.join(os.getcwd(), 'output/logs/rewards_AP_2-regualr-transitions_logs')
# # episode_evaluate(logs_file_path)
# draw_episodes(logs_file_path)

f = open('minecart.pkl', 'rb')
inf = pickle.load(f)
print(inf[0.98])
print(OPT_R)
