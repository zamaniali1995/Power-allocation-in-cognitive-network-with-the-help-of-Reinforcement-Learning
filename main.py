from power_control import GameState
from DQN import BrainDQN
import numpy as np
import matplotlib.pyplot as plt
import random as rd

PU_powers = [round(3*i/2.0,2) for i in range(1,9)]
SU_powers = [round(0.5*i/2.0,2) for i in range(1,9)]
actions = len(SU_powers)

PU_num = 5
# rd.randint(1, 10)
SU_num = 5
# rd.randint(1, 10)

Loss = []
Success = []
Fre = []

noise = 3      
# sensor_num = 10  # N 
policy = 1       # choose power change policy for PU, it should be 1(Multi-step) or 2(Single step)
brain = []
for _ in range(SU_num):
    brain.append(BrainDQN(actions,PU_num, SU_num))
com = GameState(PU_powers, SU_powers, noise, PU_num, SU_num)
terminal = True
recording = 100000

while(recording>0):    
    # initialization
    if(terminal == True):
        com.ini()
        observation0, reward0, terminal = com.frame_step(np.zeros(actions), policy, False)
        for i in range(SU_num):
            brain[i].setInitState(observation0)
    action_list = []
    loss_list = []
    # train
    for i in range(SU_num):
        action, recording = brain[i].getAction()
        action_list.append(action)
    # print(action_list)
    nextObservation, reward, terminal = com.frame_step(action_list, policy, True)
    for i in range(SU_num):
        loss_list.append(brain[i].setPerception(nextObservation, action_list[i], reward))
    loss = sum(loss_list)/len(loss_list)

    # test
    if (recording+1)%500==0:
        print("**********************************************") 
        Loss.append(loss)
        print("iteration : {} , loss : {} ." .format(100000-recording, loss)) 
        
        # print "iteration : %d , loss : %f ." %(100000-recording, loss)
        
        success = 0.0
        fre = 0
        num = 1000.0
        for ind in range(1000):
            T = 0
            com.ini_test()
            observation0_test, reward_test, terminal_test = com.frame_step_test(np.zeros(actions),policy,False)

            while (terminal_test != True) and T<20:
                action_test_list = []
                for i in range(SU_num):
                    action_test = brain[i].getAction_test(observation0_test)
                    action_test_list.append(action_test)
                observation0_test, reward_test, terminal_test = com.frame_step_test(action_test_list, policy, True)
                T +=1
            if terminal_test==True:
                success +=1
                fre +=T
        if success == 0:
            fre = 0
        else:
            fre = fre/success
        success = success/num
        Success.append(success)
        Fre.append(fre)
        print ("success : {} , step : {} ." .format(success, fre))
        print('reward: {}'.format(reward))
        
plt.plot(Loss)
plt.show()

plt.plot(Success)
plt.show()

plt.plot(Fre)
plt.show()
