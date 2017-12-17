#####################################################################################################################
########################################## Parameters ###############################################################
#  Action Set              : (None,U,D,L,R,Stay);(None,U,UL,UR,D,DL,DR,L,R,Stay);(All);(Restrictive)
#  Goal set                : (All);(One Goal);(Multi Goal)
#  Time Penalty            : Time_Penalty
#  Movement Penalty        : Movement_Penalty (Fixed_Penalty +  Running_Penalty)
#  Reward                  : (positive or negative)
#  Maze                    : (m*n)
#  Player Position         : Player_pos
#  Exchange                : Exchange
#  Initial Value           : Ini_val
#  Type                    : (Continuous or Fixed End)
#  Safe house              : Safe
#  Reward Earned           : Total_Reward
#####################################################################################################################

#####################################################################################################################
########################################## Design ###################################################################
#  Solver                  : Dense Neural Net
#  Solver activation       : Relu
#  Solver Algorithm        : Q-Learning
#  Maze Generator          : Dense Neural Net or Based on Real Time Data
#  Maze activation         : Sigmoid / Clipped Relu
#  Algorithm               : Q-Learning
#  Epsilon                 : epsilon
#  Gamma                   : gamma
#  Alpha                   : alpha
#####################################################################################################################
#%matplotlib notebook
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import random
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.layers import LSTM
from keras.utils import plot_model


#####################################################################################################################
# Initialization
logfile = open('./log.txt','w+')
# wrong action counter
wa=0
epsilon = 0.3

alpha = 0.01
gamma = 0.9/alpha
episode = 4000

buff_size = 100
# Initial Funds in cents
Ini_value = 100000
max_value = 1000
# wrong action penalty
wrong_action = max_value/100
# Maze
# maze size : m*n
maze_size = [5,5]
maze = np.random.randint(max_value/2,size=(maze_size[0],maze_size[1])).astype('float')+1500.0
#maze = 1000 - maze
Game_termination = False
#Action_set = np.random.randint(2,size=((maze_size[0]*maze_size[1])+1,maze_size[0],maze_size[1]))
Action_set = np.zeros(((maze_size[0]*maze_size[1])+1,maze_size[0],maze_size[1]))+1
#print(Action_set)
#print(Action_set[:,:,9])

# Time penalty
Time_Penalty = 5
# Movement Penalty
Fixed_Penalty = 10
Running_Penalty = 0.005*maze
Movement_Penalty = Fixed_Penalty + Running_Penalty
# Goal state when goal states are defined and maze has fixed termination
Goal_state = False
Goal_set = np.zeros((maze_size[0],maze_size[1]))

# Player location
Player_data = np.zeros((maze_size[0],maze_size[1]))
Player_pos = [random.randint(0,maze_size[0]-1),random.randint(0,maze_size[1]-1)]
Player_data[Player_pos[0],Player_pos[1]] = 1
# Reward Information
Reward_data = np.zeros((maze_size[0],maze_size[1])).astype('float')
Safe = False
# Display Maze
print(Player_data)
fig = plt.figure(frameon='False')
plt.imshow(maze,cmap = "gray",interpolation = "nearest")
#plt.colorbar()
#plt.figure()
plt.imshow(Player_data,cmap = "hot",alpha = 0.5,interpolation = "bicubic")
plt.title('Maze and player location')
#plt.colorbar()
fig.savefig('./Maze.png')
#####################################################################################################################
# Models
#####################################################################################################################
# Solver Buy
input_size = maze.size+Player_data.size
output_size = (maze_size[0]*maze_size[1])+1

maze_solver = Sequential()
maze_solver.add(Dense(input_size, activation='relu', input_dim=input_size))
maze_solver.add(Dense(input_size, activation='relu'))
maze_solver.add(Dense(output_size, activation='relu'))
maze_solver.compile(optimizer='rmsprop',
              loss='mse',
              metrics=['accuracy'])
print('Solver Model Created')
#####################################################################################################################
# Solver Sell
inputs_size = maze.size+Reward_data.size+1
outputs_size = maze_size[0]*maze_size[1]

maze_solver_s = Sequential()
maze_solver_s.add(Dense(inputs_size, activation='relu', input_dim=inputs_size))
maze_solver_s.add(Dense(inputs_size, activation='relu'))
maze_solver_s.add(Dense(outputs_size, activation='relu'))
maze_solver_s.compile(optimizer='rmsprop',
              loss='mse',
              metrics=['accuracy'])
print('Solver Model S Created')
#####################################################################################################################
# Creator
creator_insize = maze.size
maze_creator = Sequential()
maze_creator.add(Dense(creator_insize, activation='relu', input_dim=input_size))
maze_creator.add(Dense(creator_insize, activation='relu'))
maze_creator.add(Dense(creator_insize, activation='relu'))
maze_creator.compile(optimizer='rmsprop',
              loss='mse',
              metrics=['accuracy'])
print('Maze Creator Model Created')
#####################################################################################################################
# Connector
connector_insize = Action_set.size
maze_connector = Sequential()
maze_connector.add(LSTM(creator_insize, return_sequences = True, input_shape=(connector_insize,1)))
maze_connector.add(LSTM(creator_insize, return_sequences = True))
maze_connector.add(LSTM(2, return_sequences = True,activation = 'softmax'))
maze_connector.compile(optimizer='rmsprop',
              loss='mse',
              metrics=['accuracy'])

print('models created')
#####################################################################################################################
#####################################################################################################################
counter = 0
scount = 0
Game_termination = False
sfc = 0
sol = []
sel = []
inp_sol = []
inp_sell = []
tag_sol = []
tag_sell = []
Reward_history = []
market = []
bmem = []
smem = []
#Reward_history.append(Ini_value)
cash = Ini_value
market.append(0)
Total_Reward = 0
safe = True
#print(Total_Reward,Reward_data)

treasure_gain = np.zeros((maze_size[0],maze_size[1]))
treasure_sold = np.zeros((maze_size[0],maze_size[1]))
time_flag = True
while  not Game_termination:
    counter += 1
    #print('g')
    logfile.write('\nCounter :'+str(counter))
    if time_flag:
        #print('a')
        time_flag = False
        wrong_action_flag = False
        rand = False
        
        prev_index = Player_pos
        # Input to solver
        solver_input = np.concatenate((maze.flatten(),Player_data.flatten())).reshape((1,maze.size+Player_data.size))
        inp_sol.append(solver_input)
        
        #print(solver_input.shape)
        # Epsilon-greedy action selection
        if np.random.rand() <= epsilon:
            #idx = np.where(Action_set[:,prev_index[0],prev_index[1]] == 1)
            #print(Action_set[:,prev_index[0],prev_index[1]])
            #print(len(idx[0]),idx)
            #pos = idx[0][random.randint(0, len(idx[0])-1)]
            #if len(idx[0]) > 1:
            #    np.random.seed(counter)
            #    pos = idx[0][random.randint(0, len(idx[0])-1)]
            #else:
            #    pos = idx[0][0]
            #print(pos)
            rand = True
            #print('rand')
            pos = np.argmin(maze)
            #print(Reward_data*maze)
            #pos = [val//maze_size[0],val%maze_size[0]]
            logfile.write('\nPosition of Buyer : Random\n')
            logfile.write(str(pos))
        else:
            # Q learning being used
            out = maze_solver.predict(solver_input)
            out_id = np.where(out == out.max())
            #print(out_id)
            if len(out_id) > 1:
                # seed being changed to increase randomness
                np.random.seed(counter)
                pos = out_id[0][random.randint(0, len(out_id[0])-1)]
            else:
                pos = out_id[0][0]
                #print(pos)
            logfile.write('\nPosition of Buyer : Q\n')
            logfile.write(str(pos))
        #print(Action_set[pos,prev_index[0],prev_index[1]])
        logfile.write('\n##########################\n')
        logfile.write('Buyer')
        logfile.write('\n##########################')
        print(pos)
        if Action_set[pos,prev_index[0],prev_index[1]] == 1:
            if pos != output_size-1:
                Player_pos = [pos//maze_size[0],pos%maze_size[0]]
                Player_data = np.zeros((maze_size[0],maze_size[1]))
                Player_data[Player_pos[0],Player_pos[1]] = 1
                safe = False
                prev_value = maze[Player_pos[0],Player_pos[1]]
                logfile.write('\nSafe: False')
                logfile.write('\nPlayer data\n')
                logfile.write(str(Player_data))
            else:
                safe = True
                logfile.write('\nSafe: True')
                sfc += 1
                prev_value = 0
                Player_data = np.zeros((maze_size[0],maze_size[1]))
                logfile.write('\nPlayer data\n')
                logfile.write(str(Player_data))
        else:
            wrong_action_flag = True
            logfile.write('\nWrong Action: True')
            pev_value = 0
        if counter > Time_Penalty:
            Reward_history.append(Reward_history[-1])
            logfile.write('\nReward History : ')
            logfile.write(str(Reward_history[-1]))
        else:
            Reward_history.append(0)
            logfile.write('\nReward History : ')
            logfile.write('0')
    ###################################################################################################################
    if counter%Time_Penalty == 0 and counter>0:
        logfile.write('\nCounter :'+str(counter))
        #print('q')
        # Wrong action flag to check if an illegal move was performed
        # at a given state as defined by Action_set (connection graph)
        time_flag = True
        if wrong_action_flag:
            wa += 1
            wrong_action_flag = False
            print('wa')
            logfile.write('\nWrong action buyer')
            #print(cash,Movement_Penalty[Player_pos[0],Player_pos[1]],wrong_action)
            # Total liquid value
            logfile.write('\nPre Cash ,Pre Total Reward\n')
            logfile.write(str(cash)+'\t'+str(Total_Reward))
            logfile.write('\nReward data\n')
            logfile.write(str(Reward_data))
            cash = cash - Movement_Penalty[Player_pos[0],Player_pos[1]] - wrong_action
            # Total assets
            Total_Reward = np.sum(Reward_data*maze) + cash
            print(cash,Total_Reward,Reward_data)
            logfile.write('\nCash , Total Reward\n')
            logfile.write(str(cash)+'\t'+str(Total_Reward))
            logfile.write('\nReward data\n')
            logfile.write(str(Reward_data))
            spos = outputs_size-1
        else:
            print('#############nwa')
            logfile.write('\nRight Action')
            if not safe:
                safe = True
                print('not safe')
                logfile.write('\nNot safe')
                logfile.write('\nCash , Total Reward\n')
                logfile.write(str(cash)+'\t'+str(Total_Reward))
                logfile.write('\nReward data\n')
                logfile.write(str(Reward_data))
                #print(cash,Movement_Penalty[Player_pos[0],Player_pos[1]],maze[Player_pos[0],Player_pos[1]])
                cash = cash - Movement_Penalty[Player_pos[0],Player_pos[1]] - maze[Player_pos[0],Player_pos[1]]
                #print(cash)
                # Selling logic
                # The one that gives max profit is sold
                #if cash <0:
                #print(maze.flatten())
                #print(Reward_data.flatten())
                #print(int(cash))
                logfile.write('\nBefore sell Cash , Before sell Total Reward\n')
                logfile.write(str(cash)+'\t\t\t'+str(Total_Reward))
                logfile.write('\nReward data\n')
                logfile.write(str(Reward_data))
                sell_input = np.append(maze.flatten(),Reward_data.flatten())
                sell_input = np.append(sell_input,int(cash)).reshape((1,-1))
                inp_sell.append(sell_input)
                logfile.write('\n##########################\n')
                logfile.write('Seller')
                logfile.write('\n##########################')
                if np.random.rand() <= epsilon:
                    print('srand')
                    logfile.write('\nRandom Sell')
                    cash_val = np.argmax(Reward_data*maze)
                    #print(Reward_data*maze)
                    cash_pos = [cash_val//maze_size[0],cash_val%maze_size[0]]
                    spos = cash_val
                    #print(spos)
                else:
                    print('sq')
                    logfile.write('\nQ value sell')
                    sout = maze_solver_s.predict(sell_input)
                    sout_id = np.where(sout == sout.max())
                    #print(sout_id)
                    if len(sout_id) > 1:
                        # seed being changed to increase randomness
                        np.random.seed(counter)
                        spos = sout_id[0][random.randint(0, len(sout_id[0])-1)]
                        #print(spos)
                    else:
                        spos = sout_id[0][0]
                        #print(spos)
                        #print(pos)   
                    #print(Action_set[pos,prev_index[0],prev_index[1]])
                #if spos != outputs_size-1:
                        
                cash_pos = [spos//maze_size[0],spos%maze_size[0]]
                logfile.write('\nPosition of Sell :'+str(cash_pos))
                if Reward_data[cash_pos[0],cash_pos[1]] == 0:
                    cash -= wrong_action
                    logfile.write('\nWrong Action Seller')

                else:
                    print(cash_pos)
                    print('sold')
                    logfile.write('\nSold')
                    scount += 1
                    print('Reward Data')
                    #print(Reward_data[cash_pos[0],cash_pos[1]]*maze[cash_pos[0],cash_pos[1]])
                    cash += Reward_data[cash_pos[0],cash_pos[1]]*maze[cash_pos[0],cash_pos[1]]
                    treasure_sold[cash_pos[0],cash_pos[1]] += Reward_data[cash_pos[0],cash_pos[1]]
                    Reward_data[cash_pos[0],cash_pos[1]] = 0
                    #print('total reward')
                    #print(Total_Reward)
                    #print(Reward_data)
                    logfile.write('\nCash , Total Reward\n')
                    logfile.write(str(cash)+'\t'+str(Total_Reward))
                    logfile.write('\nReward data\n')
                    logfile.write(str(Reward_data))
                #############################################################################################################    
                # Seller
                # experience replay one experience
                targets = maze_solver_s.predict(sell_input).flatten()
                #print(targets)
                # Next state
                sell_input1 = np.append(maze.flatten(),Reward_data.flatten())
                sell_input1 = np.append(sell_input1,int(cash)).reshape((1,-1))
                sout1 = maze_solver_s.predict(sell_input1)
                sout_id1 = np.where(sout1 == sout1.max())
                print(sout_id1)
                if len(sout_id1[0]) > 1:
                    spos1 = sout_id1[0][random.randint(0, len(sout_id1[0])-1)]
                else:
                    spos1 = sout_id1[0][0]
                #print(out1[0][pos1])
                # Accounting for future reward
                #print(spos)
                targets[spos] = alpha*(Total_Reward-Reward_history[-1] + (gamma*sout1[0][spos1]))
                tag_sell.append(targets.flatten())
                sel.append([sell_input,targets])
                #print(inp_sol,tag_sol)
                if counter >2:
                    # Online training
                    #maze_solver_s.fit(sell_input,targets.reshape((1,outputs_size)))
                    maze_solver_s.fit(np.asarray(inp_sell).reshape((-1,inputs_size)),\
                                      np.asarray(tag_sell).reshape((-1,outputs_size)),epochs=20)
                    #Action_set = 1-Action_set                


            Reward_data[Player_pos[0],Player_pos[1]] += 1
            treasure_gain[Player_pos[0],Player_pos[1]] += 1    
            Total_Reward = np.sum(Reward_data*maze) + cash
            #print('total reward')
            #print(Total_Reward)
            #print(cash,Total_Reward,Reward_data*maze)
            logfile.write('\nAfter sell')
            logfile.write('\nCash , Total Reward\n')
            logfile.write(str(cash)+'\t'+str(Total_Reward))
            logfile.write('\nReward data\n')
            logfile.write(str(Reward_data))


        Reward_history.append(Total_Reward.flatten())
        logfile.write('\nReward History : ')
        logfile.write(str(Total_Reward.flatten()))
        ##############################################################################################################
        # Buyer
        # experience replay one experience
        #print('total reward')
        #print(Total_Reward)
        target = maze_solver.predict(solver_input).flatten()
        #print(target)
        # Next state
        solver_input1 = np.concatenate((maze.flatten(),Player_data.flatten())).reshape((1,maze.size+Player_data.size))
        out1 = maze_solver.predict(solver_input1)
        out_id1 = np.where(out1 == out1.max())
        #print(out_id1)
        if len(out_id1) > 1:
            pos1 = out_id1[0][random.randint(0, len(out_id1[0])-1)]
        else:
            pos1 = out_id1[0][0]
        #print(out1[0][pos1])
        # Accounting for future reward
        target[pos] = alpha*(-Total_Reward+Reward_history[-1] + (gamma*out1[0][pos1]))
        tag_sol.append(target.flatten())
        sol.append([solver_input,target])
        #print('target')
        #print(target[pos])
        #print(target)
        #print(inp_sol,tag_sol)
        if counter >2:
            # Online training
            #maze_solver.fit(solver_input,target.reshape((1,output_size)))
            maze_solver.fit(np.asarray(inp_sol).reshape((-1,input_size)),\
                                      np.asarray(tag_sol).reshape((-1,output_size)),epochs=20)
            #Action_set = 1-Action_set
        
        #epsilon *= 0.999
    ##############################################################################################################
    mk = np.random.rand()
    #print('m')
    # Maze Creator
    
    if mk <= 0.5:
        market.append(-1)
        np.random.seed(counter*100)
        maze = np.random.randint(max_value/2,size=(maze_size[0],maze_size[1]))+1500
        #Action_set = np.random.randint(2,size=((maze_size[0]*maze_size[1])+1,maze_size[0],maze_size[1]))
    
    elif mk >= 0.9:
        market.append(1)
        np.random.seed(counter*100)
        maze = max_value - np.random.randint(max_value,size=(maze_size[0],maze_size[1]))+1510
        #maze_creator.predict(maze.reshape(1,creator_insize))
        #Action_set = 1-np.random.randint(2,size=((maze_size[0]*maze_size[1])+1,maze_size[0],maze_size[1]))
    
    else:
        market.append(0)
        np.random.seed(counter*100)
        maze = np.random.randint(max_value,size=(maze_size[0],maze_size[1]))+1490
        #maze_creator.predict(maze.reshape(1,creator_insize))
        #Action_set = 1-np.random.randint(2,size=((maze_size[0]*maze_size[1])+1,maze_size[0],maze_size[1]))
    #print('Input')
    #print(len(inp_sol))
    #print('Target')
    #print(len(tag_sol))
    #if len(inp_sol)>buff_size:
    #    del inp_sol[0]
    if len(tag_sol)>buff_size:
        del inp_sol[0]
        del tag_sol[0]
    #if len(inp_sell)>buff_size:
    #    del inp_sell[0]
    if len(tag_sell)>buff_size:
        del inp_sell[0]
        del tag_sell[0]
    #if len(sol)>buff_size:
    #    del sol[0]
    #if len(sel)>buff_size:
    #    del sel[0] 
    
    logfile.write('\nCounter maze :'+str(counter))
    if counter > Time_Penalty and not time_flag and counter%Time_Penalty != 0:
        Reward_history.append(Reward_history[-1])
        logfile.write('\nReward History : ')
        logfile.write(str(Reward_history[-1]))
    elif counter <= Time_Penalty and not time_flag:
        Reward_history.append(0)
    logfile.write('\nMaze creation\n')
    logfile.write(str(maze))
    logfile.write('\nCash , Total Reward\n')
    logfile.write(str(cash)+'\t'+str(Total_Reward))
    logfile.write('\nReward data\n')
    logfile.write(str(Reward_data))
    
    Fixed_Penalty = 10
    Running_Penalty = 0.005*maze
    Movement_Penalty = Fixed_Penalty + Running_Penalty
    
    logfile.write('\nMovement Penalty\n')
    logfile.write(str(Movement_Penalty))
    logfile.write('\n##########################################################################################\n')
    if counter>episode:
        Game_termination = True
#rw = np.array(Reward_history)
#print(rw)
print(scount,sfc,wa,counter)
print(Reward_history[-1])
print(treasure_gain)
print(treasure_sold)
#print(inp_sol)
#print(tag_sol)
#print(np.asarray(sol[0:3][0]).shape)
#print(solver_input.size)
x = np.linspace(0,episode,num=len(np.array(Reward_history)))
z = np.zeros(shape=(len(x)))
#print(len(x))
#print(Reward_history)
fig3=plt.figure()
#plt.plot(x,np.array(Reward_history))
plt.plot(x,np.array(Reward_history),'g')
plt.plot(x,z,'r')
#plt.plot(x[np.array(Reward_history)<0],np.array(Reward_history)[np.array(Reward_history)<0],'r')
plt.ylabel('Total Asset')
plt.xlabel('Number of Moves')
plt.title('Generation of Asset over time')
plt.show()
fig3.savefig('./asset.png')


