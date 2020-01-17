import random
import numpy as np
import math

class GameState:
    def __init__(self, PU_powers, SU_powers, Noise, PU_num, SU_num):
        
        self.PU_powers = PU_powers
        self.SU_powers = SU_powers
        self.length_PU_powers = len(self.PU_powers)
        self.length_SU_powers = len(self.SU_powers)
        
        self.h_11 = 1;self.h_12 = 1
        self.h_21 = 1;self.h_22 = 1
        
        self.ita_PU = 0.1
        self.ita_SU = 0.08
        
        self.sigma_sq_1 = 0.01
        self.sigma_sq_2 = 0.01
        
        self.lam = 0.1
        self.alpha = 0.5

        self.PN_dimension = 300
        
        self.SN_diemension = 50

        self.PT_position_x = self.PN_dimension/2
        self.PT_position_y = self.PN_dimension/2
        self.ST_position_x = random.randint(0, self.PN_dimension)
        self.ST_position_y = random.randint(0, self.PN_dimension)

        self.PU_num = PU_num
        self.SU_num = SU_num

        self.noise = Noise
        
        self.P_SS, self.P_SP, self.P_PP ,self.sigma = self.dis()
        
    def dis(self):
        dis_SU = np.random.uniform(low=0, high=self.SN_diemension, size=(self.SU_num))
        dis_PU = np.random.uniform(low=0, high=self.PN_dimension, size=(self.PU_num))
        
        ang_SU = np.random.uniform(low=0, high=360, size=(self.SU_num))
        ang_PU = np.random.uniform(low=0, high=360, size=(self.PU_num))
        
        P_SS = np.zeros(shape=(self.SU_num, self.SU_num), dtype=float)
        P_SP = np.zeros(shape=(self.SU_num, self.PU_num), dtype=float)
        P_PP = np.zeros(shape=(self.PU_num, self.PU_num), dtype=float)

        for i in range(self.SU_num):
            for j in range(self.SU_num):
                if i != j:
                    P_SS[i][j] = (self.lam/(4*math.pi)/self.cal_dis(dis_SU, ang_SU,S_1=i, S_2=j))**2
                else:
                    P_SS[i][j] = (self.lam/(4*math.pi)/dis_SU[i])**2

        for i in range(self.SU_num):
            for j in range(self.PU_num):
                P_SP[i][j] = (self.lam/(4*math.pi)
                    /self.cal_dis(dis_SU, ang_SU, dis_PU, ang_PU, S_1=i, P=j))**2 

        for i in range(self.PU_num):
            for j in range(self.PU_num):
                if i != j:
                    P_PP[i][j] = (self.lam/(4*math.pi)/self.cal_dis(dis_PU, ang_PU,S_1=i, S_2=j))**2
                else:
                    P_PP[i][j] = (self.lam/(4*math.pi)/dis_PU[i])**2
        
        sigma = np.zeros((self.SU_num))
        for i in range(self.SU_num):
            for s in range(self.SU_num):
                sigma[i] += P_SS[i][s] * self.SU_powers[0]
            for p in range(self.PU_num):
                sigma[i] += P_SP[i][p] * self.PU_powers[0]
            sigma[i] /= self.noise
        return P_SS, P_SP, P_PP ,sigma

    def cal_dis(self, dis_SU=None, ang_SU=None, dis_PU=None, ang_PU=None, S_1=None, S_2=None, P=None):
        if S_1 != None and S_2 != None:
            ang = abs(ang_SU[S_1]-ang_SU[S_2])
            if ang > 180:
                ang -= 180
            if ang == 0:
                return abs(dis_SU[S_1]-dis_SU[S_2])
            elif ang == 180:
                return dis_SU[S_1]+dis_SU[S_2]
            return dis_SU[S_1]**2+dis_SU[S_2]**2-2*dis_SU[S_1]*dis_SU[S_2]*math.cos(ang)
        
        elif S_1 != None and P != None:
            SU_x = self.ST_position_x+dis_SU[S_1]*math.cos(dis_SU[S_1])
            SU_y = self.ST_position_x+dis_SU[S_1]*math.sin(dis_SU[S_1])

            PU_x = self.PT_position_x+dis_PU[P]*math.cos(ang_PU[P])
            PU_y = self.PT_position_x+dis_PU[P]*math.sin(ang_PU[P])

            return math.sqrt(abs(SU_x-PU_x)**2 + abs(SU_y-PU_y)**2)
            
    def ini(self):
        self.SU_power = np.zeros((self.SU_num))
        self.PU_power = np.zeros((self.PU_num))
        for i in range(self.SU_num):
            self.SU_power[i] = self.SU_powers[random.randint(0, self.length_SU_powers-1)]
        for i in range(self.PU_num):
            self.PU_power[i] = self.PU_powers[random.randint(0, self.length_PU_powers-1)]    
        
    def ini_test(self):
        self.SU_power_test = np.zeros((self.SU_num))
        self.PU_power_test = np.zeros((self.PU_num))
        for i in range(self.SU_num):
            self.SU_power_test[i] = self.SU_powers[random.randint(0, self.length_SU_powers-1)]
        for i in range(self.PU_num):
            self.PU_power_test[i] = self.PU_powers[random.randint(0, self.length_PU_powers-1)]    
      
    def frame_step(self, input_actions, policy, i):
        if i == True:
            if policy == 1:
                self.PU_power = self.update_p1_v1(self.SU_power, self.PU_power)
            if policy == 2:
                self.PU_power = self.update_p1_v2(self.PU_power, self.SU_power)
            for i, action in enumerate(input_actions):
                self.SU_power[i] = self.SU_powers[action]
                
        observation = self.compute_observation(self.PU_power, self.SU_power)
        reward = self.compute_reward(self.PU_power, self.SU_power)
        
        terminal = (reward==self.SU_num+3*self.PU_num)
        
        return observation,reward,terminal

    def frame_step_test(self, input_actions, policy, i):
        if i == True:
            if policy == 1:
                self.PU_power_test = self.update_p1_v1(self.SU_power_test, self.PU_power_test)
            if policy == 2:
                self.p_1_test = self.update_p1_v2(self.p_1_test,self.p_2_test)
            for i, action in enumerate(input_actions):
                self.SU_power_test[i] = self.SU_powers[action]
            
        observation = self.compute_observation(self.PU_power, self.SU_power)
        reward = self.compute_reward(self.PU_power_test, self.SU_power_test)
        
        terminal = (reward==self.SU_num+3*self.PU_num) 
        
        return observation,reward,terminal
    
    def compute_observation(self, PU_power, SU_power): 
        observation = np.zeros((self.SU_num))
        for i in range(self.SU_num):
            for s in range(self.SU_num):
                observation[i] += self.P_SS[i][s] * SU_power[s]
            for p in range(self.PU_num):
                observation[i] += self.P_SP[i][p] * PU_power[p]
            observation += random.gauss(0,self.sigma[i])
            if observation[i] < 0:
                observation[i] = 0
            observation[i] = observation[i]*(10**7)
        return observation
    
    def compute_reward(self, PU_power, SU_power):
        PU_success, SU_success = self.compute_SINR(PU_power, SU_power)
        reward = 3*sum(PU_success)+sum(SU_success)
        return reward
    
    def update_p1_v1(self, SU_power, PU_power):
        for p_1 in range(self.PU_num):
            p_1_n = self.ita_PU/abs(self.h_11)**2
            tmp_2 = 0
            for s in range(self.SU_num):
                tmp_2 += (abs(self.h_21)**2)*SU_power[s]
            for p_2 in range(self.PU_num):
                if p_1 != p_2:
                    tmp_2 += (abs(self.h_21)**2)*PU_power[p_1]
            tmp_2 += self.sigma_sq_1
            p_1_n = p_1_n/tmp_2
            # tmp_1 = 0
            tmp_2 = 0
            v = []
            for ind in range(self.length_PU_powers):
                v.append(max(p_1_n-self.PU_powers[ind],0))
            PU_power[p_1] = self.PU_powers[v.index(min(v))]
        return PU_power
    
    def update_p1_v2(self,x,y):
        ind_p_1 = self.P_1.index(x)
        tSINR_1 = ((abs(self.h_11)**2)*x/((abs(self.h_21)**2)*y + self.sigma_sq_1))
        tao = x * self.ita_1 / tSINR_1
        if tao>=x and ind_p_1+1<=self.length_P_1-1 and tao<=self.P_1[ind_p_1+1] :
            x = self.P_1[ind_p_1+1]
        elif ind_p_1-1>=0 and tao<=self.P_1[ind_p_1-1] :
            x = self.P_1[ind_p_1-1]
        return x
    
    def compute_SINR(self, PU_power, SU_power):
        PU_success = np.zeros(self.PU_num)
        SU_success = np.zeros(self.SU_num)
        for p_1 in range(self.PU_num):
            PU_success[p_1] = (abs(self.h_11)**2)*PU_power[p_1]
            tmp = 0
            for s in range(self.SU_num):
                tmp += (abs(self.h_12)**2)*SU_power[s]
            for p_2 in range(self.PU_num):
                if p_1 != p_2:
                    tmp += (abs(self.h_21)**2)*PU_power[p_2]
            tmp += self.sigma_sq_1
            PU_success[p_1] /= tmp
            
            PU_success[p_1] = PU_success[p_1] >= self.ita_PU
        for s_1 in range(self.SU_num):
            SU_success[s_1] = (abs(self.h_22)**2)*self.SU_power[s_1]
            tmp = 0
            for s_2 in range(self.SU_num):
                if s_1 != s_2:
                    tmp += (abs(self.h_21)**2)*SU_power[s_2]
            for p in range(self.PU_num):
                tmp += (abs(self.h_21)**2)*PU_power[p]
            tmp += self.sigma_sq_2
            SU_success[s_1] /= tmp
            SU_success[s_1] = SU_success[s_1] >= self.ita_SU
        return PU_success, SU_success
