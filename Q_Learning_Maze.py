#MTRE-6400 Group 3 Project 4 - Randy Aboona, Michael Bakareke, Efosa Ogiesoba, Saif Sabti

# Import standard python libraries
import numpy as np
import matplotlib.pyplot as plt

# The maze's virtual environment matrix
maze = np.array([[1, 2, 3, 4],
                 [5, 6, 7, 8],
                 [9, 10, 11, 12],
                 [13, 14, 15, 16]])

# Define the state transition matrix (The maze's virtual environment)
newState = np.array([[1, 2, 1, 5],
                     [1, 3, 2, 6],
                     [2, 4, 3, 7],
                     [3, 4, 4, 8],
                     [5, 6, 1, 9],
                     [5, 7, 2, 10],
                     [6, 8, 3, 11],
                     [7, 8, 4, 12],
                     [9, 10, 5, 13],
                     [9, 11, 6, 14],
                     [10, 12, 7, 15],
                     [11, 12, 8, 16],
                     [13, 14, 9, 13],
                     [13, 15, 10, 14],
                     [14, 16, 11, 15],
                     [15, 16, 12, 16]])

# Define Q table, initialize all Q values to zero.
# There are 16 states and 4 actions.
Q = np.zeros((16, 4))

# Initialize the parameters
tao = 0.99  # the temperature parameter
ata = 0.2  # the learning rate
beta = 0.4  # the discount factor

# The initial state is State #1.
s = 0
s_history = [s]  # record the current state in s_history

i = 1
Q_history = []  # Initialize Q_history list to store Q table at each step
while i <= 50000:  # 5000 time steps
    # Calculate the probability for each action in the current state
    sum_exp = sum(np.exp(Q[s] / tao))
    P = np.exp(Q[s] / tao) / sum_exp * 100  # P(a) is the probability of action a.

    # Probabilistically select an action with P(1), P(2), P(3) and (P3)
    ran = 100 * np.random.rand()  # generate a random number r, 0 < r < 100.

    if ran <= P[0]:
        a = 0  # the first action is selected
    elif P[0] < ran <= (P[0] + P[1]):
        a = 1  # the second action is selected
    elif (P[0] + P[1]) < ran <= (P[0] + P[1] + P[2]):
        a = 2  # the third action is selected
    else:
        a = 3  # the fourth action is selected

    # Determine the next state using the state transition matrix
    # This incorporates a probablity state transion function.
    ran = 100 * np.random.rand()  # generate a random number r, 0 < r < 100.
    desired_state_prob = 100 # 100 is 100% chance (no prob state trans. func.) and 80 is 80% chance (includes prob state trans func.)
    if ran <= desired_state_prob:
        newS = newState[s, a] - 1  # The desired state (s, a)
    elif 80 < ran <= 90:
        # Transition to a state perpendicular to the action/movement
        if a == 0:  # Move left
            newS = newState[s, 2] - 1  # Move up
        elif a == 1:  # Move right
            newS = newState[s, 3] - 1  # Move down
        elif a == 2:  # Move up
            newS = newState[s, 0] - 1  # Move left
        else:  # Move down
            newS = newState[s, 1] - 1  # Move right
    else:
        # Transition to the other side perpendicular to the action/movement
        if a == 0:  # Move left
            newS = newState[s, 3] - 1  # Move down
        elif a == 1:  # Move right
            newS = newState[s, 2] - 1  # Move up
        elif a == 2:  # Move up
            newS = newState[s, 1] - 1  # Move right
        else:  # Move down
            newS = newState[s, 0] - 1  # Move left

    # Determine the immediate reward
    if newS == 11:
        r = 20
    elif newS == 6:
        r = -10
    elif newS == 13:
        r = -10
    else:
        r = 0

    # Find the maximum for all Q values corresponding to the new state
    maxQ = np.max(Q[newS])

    # Update the Q value
    Q[s, a] = (1 - ata) * Q[s, a] + ata * (r + beta * maxQ)

    # Add the new state in s_history
    s_history.append(newS)

    # Update the current state
    if newS == 11:
        print("---------------------------------")
        print("The goal is reached! The path is:")
        # Add 1 to each state value in s_history to reflect the actual state numbers based on the robot grid
        s_history_actual = [state + 1 for state in s_history]
        print(s_history_actual)

        print("")
        print("Return to the start position now")
        s_history = [0]  # empty s_history

        # Return the start position
        s = 0
    else:
        s = newS

    # Update the tao parameter
    tao *= 0.999
    if tao < 0.01:
        tao = 0.01

    # Save the Q table in Q_history
    Q_history.append(Q.copy())  # Append a copy of Q table to Q_history
    i += 1

# Plot the history of Q(9,1), Q(9,2), Q(9,3) and Q(9,4)
len_history = len(Q_history)
Q_9_1 = np.zeros(len_history)
Q_9_2 = np.zeros(len_history)
Q_9_3 = np.zeros(len_history)
Q_9_4 = np.zeros(len_history)

for j in range(len_history):
    Q_9_1[j] = Q_history[j][8, 0]
    Q_9_2[j] = Q_history[j][8, 1]
    Q_9_3[j] = Q_history[j][8, 2]
    Q_9_4[j] = Q_history[j][8, 3]

# History of Q values for State 9
plt.plot(Q_9_1, label='Q(9,1)')
plt.plot(Q_9_2, label='Q(9,2)')
plt.plot(Q_9_3, label='Q(9,3)')
plt.plot(Q_9_4, label='Q(9,4)')
plt.legend()
plt.title('History of Q values for State 9')
plt.show()

# Define colors for each state/grid
colors = ['blue', 'green', 'orange', 'purple',
          'cyan', 'yellow', 'black', 'brown',
          'red', 'lightblue', 'lightgreen', 'lightyellow',
          'lightgrey', 'black', 'lightsalmon', 'lightpink']

# Visualize the path on the grid with each state having its own color
plt.figure()
for i in range(4):
    for j in range(4):
        plt.text(j + 0.5, i + 0.5, str(maze[i, j]), ha='center', va='center', color='black', fontsize=12)
        plt.fill_between([j, j + 1], i, i + 1, color=colors[maze[i, j] - 1], alpha=0.4)

# Highlight the path
path_x = [(p - 1) % 4 + 0.5 for p in s_history_actual]
path_y = [(p - 1) // 4 + 0.5 for p in s_history_actual]
plt.plot(path_x, path_y, color='red', linewidth=2)
plt.title('Robot Path on Grid')
plt.xticks(range(5), range(1, 6))
plt.yticks(range(5), range(1, 6))
plt.gca().invert_yaxis()
plt.show()