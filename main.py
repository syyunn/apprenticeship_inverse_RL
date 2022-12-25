import gym
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from IPython.display import clear_output
import pickle
import time
import sys
import math
import seaborn as sns
import pathlib

nbins = 10 
GAMMA = 0.9
ALPHA = 0.01

def max_dict(d): 
    """
    looking for the action that gives the maximum value for a given state
    """
    max_v = float('-inf')
    for key, val in d.items():
        if val > max_v:
            max_v = val
            max_key = key
    return max_key, max_v

def create_bins(): 
    """
    create bins to discretize the continuous observable state space
    """
    # obs[0] -> cart position --- -4.8 - 4.8
    # obs[1] -> cart velocity --- -inf - inf
    # obs[2] -> pole angle    --- -41.8 - 41.8
    # obs[3] -> pole velocity --- -inf - inf

    bins = np.zeros((4,nbins))
    bins[0] = np.linspace(-4.8, 4.8, nbins)
    bins[1] = np.linspace(-5, 5, nbins)
    bins[2] = np.linspace(-.418, .418, nbins)
    bins[3] = np.linspace(-5, 5, nbins)
    return bins

def assign_bins(observation, bins): 
    """
    discretizing the continuous observation space into state
    """
    state = np.zeros(4)
    for i in range(4):
        state[i] = np.digitize(observation[i], bins[i])
    return state

def get_state_as_string(state):
    """
    encoding the state into string as dictionary
    """
    string_state=''
    for e in state:
            string_state = string_state+str(int(e)).zfill(2)
    return string_state

def get_all_states_as_string():
    states = []
    for i in range (nbins+1):
        for j in range (nbins+1):
            for k in range(nbins+1):
                for l in range(nbins+1):
                    a=str(i).zfill(2)+str(j).zfill(2)+str(k).zfill(2)+str(l).zfill(2)
                    states.append(a)
    return states

def initialize_Q():
    """
    initialize your Q table
    """
    Q = {}

    all_states = get_all_states_as_string()
    for state in all_states:
        Q[state] = {}
        for action in range(env.action_space.n):
            Q[state][action] = 0
    return Q

def play_one_game(bins, Q, eps=0.5):
    """
    train 1 episode
    """
    observation, _ = env.reset() # +is info.
    done = False
    cnt = 0 # number of moves in an episode
    state = get_state_as_string(assign_bins(observation, bins))
    total_reward = 0

    while not done:
        cnt += 1
        # np.random.randn() seems to yield a random action 50% of the time ?
        if np.random.uniform() < eps:
            act = env.action_space.sample() # epsilon greedy
        else:
            act = max_dict(Q[state])[0]

        observation, reward, done, _, _ = env.step(act)

        total_reward += reward

        if done and cnt < 200:
            reward = -300

        state_new = get_state_as_string(assign_bins(observation, bins))

        a1, max_q_s1a1 = max_dict(Q[state_new])
        Q[state][act] += ALPHA*(reward + GAMMA*max_q_s1a1 - Q[state][act])
        state, act = state_new, a1

    return total_reward, cnt

def play_many_games(bins, N=10000):
    """
    train many episodes
    """
    Q = initialize_Q()

    length = []
    reward = []
    for n in range(N):
        #eps=0.5/(1+n*10e-3)
        eps = 1.0 / np.sqrt(n+1)

        episode_reward, episode_length= play_one_game(bins, Q, eps)

        if n % 100 == 0:
            clear_output(wait=True)
            print("Episode: %d, Epislon: %.4f, Reward %d"%(n,eps,episode_reward))
        length.append(episode_length)
        reward.append(episode_reward)
    env.close()
    return length, reward, Q

def plot_running_avg(totalrewards,title='Running Average',save=False,name='result'):
    """
    plotting the average reward during training
    """
    fig=plt.figure()
    N = len(totalrewards)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = np.mean(totalrewards[max(0, t-100):(t+1)])
    plt.plot(running_avg)
    plt.title(title)
    plt.xlabel("Iterations")
    plt.ylabel("Reward")
    plt.grid()
    if save:
        plt.savefig(name+'.png',bbox_inches='tight')
    else:
        plt.show()
    
def play_policy(bins,Q,N=1000,render=False,delay=0.01):
    """
    run an environment using a trained policy
    """
    
    totalReward=[]
    steps=[]
    for n in range(N):
        print(n)
        observation, _ = env.reset()
        done=False
        episodeReward=0
        while not done:
            if render:
                env.render()
                time.sleep(delay)
            state=get_state_as_string(assign_bins(observation, bins))
            act=max_dict(Q[state])[0]
#             print(act)
            observation,reward,done,_,_=env.step(act)
            episodeReward+=reward
        totalReward.append(episodeReward)
    env.close()
    return totalReward

bins = create_bins()

env = gym.make('CartPole-v0')
# episode_lengths, episode_rewards, expert_Q=play_many_games(bins,N=3000)
# plot_running_avg(episode_rewards)

# save trained expert model
# print("export trained expert model...")
# filename = 'expert_Q'
# outfile = open(filename,'wb')
# pickle.dump(expert_Q,outfile)
# outfile.close()

with open('expert_Q', 'rb') as f: 
    expert_Q = pickle.load(f)

# plot reward distribution for all episodes 
# expertReward=play_policy(bins,expert_Q, N=1000, render=False)
# plt.hist(expertReward,bins=50)
# plt.title("Reward Distribution")
# plt.xlabel("Reward")
# plt.ylabel("Frequency")
# plt.show()

# build IRL algorithm
def sigmoid(arry):
    sig=[]
    for i in arry:
        sig.append(1/(1+math.exp(-i)))           
    return np.array(sig)

def getFeatureExpectation(Q,N=1000): # get estimated feature expectation by plug-in method
    # N is the number of episodes
    observationSum=np.zeros(4)
    for i in range(N):
        observation,_=env.reset()
        done=False
        cnt=0
        while not done:
            state=get_state_as_string(assign_bins(observation, bins))
            act=max_dict(Q[state])[0]
            observation, reward, done, _, _ =env.step(act)
            observation=sigmoid(observation) # take sigmoid for each dimension of observation
            observationSum+=(GAMMA**cnt)*observation
            cnt+=1
    featureExpectation=observationSum/N # equation 5 in the paper
    
    print("FeatureExpectation: ",featureExpectation)
    return featureExpectation

def irl_play_one_game(bins,weight,Q,eps=0.5):
    observation, _ = env.reset()
    done = False
    cnt = 0 # number of moves in an episode
    state = get_state_as_string(assign_bins(observation, bins))
    total_reward = 0
    

    while not done:
        cnt += 1
        # np.random.randn() seems to yield a random action 50% of the time ?
        if np.random.uniform() < eps:
            act = env.action_space.sample() # epsilon greedy
        else:
            act = max_dict(Q[state])[0]

        observation, reward, done, _, _ = env.step(act)
        
        #encode observations into state
        state_new = get_state_as_string(assign_bins(observation, bins))
        
        #map observations to 0 and 1
        observation=sigmoid(observation)
        
        #discard the simulation reward, and use the reward function found from irl algorithm
        reward = np.dot(weight,observation)

        total_reward += reward

        if done and cnt < 200:
            reward = -1


        a1, max_q_s1a1 = max_dict(Q[state_new])
        Q[state][act] += ALPHA*(reward + GAMMA*max_q_s1a1 - Q[state][act])
        state, act = state_new, a1

    return total_reward, cnt

def irl_play_many_games(bins,weight,N=10000):
    Q = initialize_Q()
    length = []
    reward = []
    for n in range(N):
        eps = 1.0 / np.sqrt(n+1)

        episode_reward, episode_length= irl_play_one_game(bins, weight,Q,eps)

        length.append(episode_length)
        reward.append(episode_reward) 
    print("Avg Length %d"%(np.average(length)))
    print("standard deviation %d"%(np.std(length)))
    return length, reward, Q

expertExpectation=getFeatureExpectation(expert_Q,N=100000) #get expert feature expectation, then use it to calculate weight.
# \mu(\pi_E) in the paper


#either terminate with margin or iteration
epislon=0.00002
N=10

weight=[]
featureExpectation=[]
featureExpectationBar=[]
learnedQ=[]
margin=[]
avgEpisodeLength=[]

for i in range(N):
    print("Iteration: ",i)
    if i==0: #step1, initialization
        initialQ=initialize_Q() #give random initial policy
        featureExpectation.append(getFeatureExpectation(initialQ))
        print("expert feature Expectation: ", expertExpectation)
        learnedQ.append(initialQ) #put in the initial policy
        weight.append(np.zeros(4)) #put in a dummy weight
        margin.append(1) #put in a dummy margin
    else:#first iter of step 2
        if i==1:
            featureExpectationBar.append(featureExpectation[i-1])
            weight.append(expertExpectation-featureExpectation[i-1])
            margin.append(norm((expertExpectation-featureExpectationBar[i-1]),2))

            print("margin: ",margin[i])
            print("weight: ",weight[i])

        else: #iter 2 and on of step 2
            A=featureExpectationBar[i-2]
            B=featureExpectation[i-1]-A
            C=expertExpectation-featureExpectationBar[i-2]
            featureExpectationBar.append(A+(np.dot(B,C)/np.dot(B,B))*(B))

            weight.append(expertExpectation-featureExpectationBar[i-1]) # update weight
            margin.append(norm((expertExpectation-featureExpectationBar[i-1]),2))

            print("margin: ",margin[i])
            print("weight: ",weight[i])
            
        #step3,terminate condition    
        if (margin[i]<=epislon):
            break

        #step4
        episode_lengths, episode_rewards, learnedQ_i= irl_play_many_games(bins,weight[i])
        learnedQ.append(learnedQ_i)
        avgEpisodeLength.append(episode_lengths)
        #step5
        featureExpectation.append(getFeatureExpectation(learnedQ[i]))
    
    print("")

print("export trained IRL model...")
filename = 'learnedQ'
outfile = open(filename,'wb')
pickle.dump(learnedQ,outfile)
outfile.close()

#showing the performance of each student
for i in range(0,len(avgEpisodeLength)):
    title="Student "+str(i+1)+" Running Average"
    plot_running_avg(avgEpisodeLength[i],title=title,save=True,name=title)

#Plotting Convergence Rate
plt.plot(margin)
plt.title("Distance To Expert Feature Distribution")
plt.xlabel("Iteration")
plt.ylabel("Distance")
plt.yscale("log")
plt.grid()
plt.savefig("Distance To Expert Feature Distribution")

#showing the performance of each student relative to the performance of the expert
iteration=[]
relativePerformance=[]
studentPerformance=[]

for i in avgEpisodeLength:
    studentPerformance.append(np.average(i))

for i in range(np.size(studentPerformance)):
    iteration.append(i)
    relativePerformance.append(studentPerformance[i]/200)
    
plt.plot(iteration,relativePerformance)
plt.xlabel("Iteration")
plt.ylabel("Relative Performance to Expert Policy")
plt.title("Training Result")
plt.grid()
plt.savefig("Distance To Expert Feature Distribution")

if __name__ == "__main__":
    pass
