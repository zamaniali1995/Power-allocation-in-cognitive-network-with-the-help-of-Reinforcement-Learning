from power_control import GameState
from DQN import BrainDQN
import numpy as np
import matplotlib.pyplot as plt
import random as rd
import csv


PU_powers = [round(i/2.0,2) for i in range(1,50)]
SU_powers = [round(i/10.0,2) for i in range(1,15)]
actions = len(SU_powers)

PU_num = 2
# rd.randint(1, 10)
SU_num = 5
# rd.randint(1, 10)

Loss = []
Success = []
Fre = []

noise = 3      
policy = 1       # choose power change policy for PU, it should be 1(Multi-step) or 2(Single step)

brain = BrainDQN(actions, PU_num, SU_num)
com = GameState(PU_powers, SU_powers, noise, PU_num, SU_num)
terminal = True
recording = 1000

while(recording>0):    
    # initialization
    if(terminal == True):
        com.ini()
        observation0, reward0, terminal = com.frame_step(np.zeros(actions), policy, False)
        brain.setInitState(observation0)
    # train
    action, recording = brain.getAction()
    nextObservation, reward, terminal = com.frame_step(action, policy, True)
    loss = brain.setPerception(nextObservation, action, reward)
    
    # test
    if (recording+1)%500==0:
        print("*"*30) 
        Loss.append(loss)
        print("iteration : {} , loss : {} ." .format(1000-recording, loss))     
        success = 0.0
        fre = 0
        num = 1000.0
        for ind in range(1000):
            T = 0
            com.ini_test()
            observation0_test, reward_test, terminal_test = com.frame_step_test(np.zeroR)
    wr = csv.writer(resultFile, dialect='excel')
    wr.writerows(map(lambda x: [x], Success))
# plt.show()

plt.plot(Fre)
plt.savefig('Fre.pdf')
plt.savefig('Fre.png')
plt.close()
with open("./Fre.csv",'w+') as resultFile:
    wr = csv.writer(resultFile, dialect='excel')
    wr.writerows(map(lambda x: [x], Fre))
# plt.show()
