# ReinforcementLearning
Solving a maze that moves in time and space with application in trading, games, speech etc
## Maze
-	Divided into multiple rooms
-	Each room is connected to every other room
-	Each room can have treasure/poison/trap etc.
-	Each room can serve as entrance and exit
-	Can be changed to from one room as entrance/exit to all room as entrance/exit
-	The maze can also be rotated along the axis
-	Adding more dimension will make it like a Rubik’s cube.
-	The value of treasure will depend on the current value of its source room

## Energy
-	The Solver has fixed energy at the beginning
-	Movement to other room decreases energy
-	Converting treasure to energy consumes timestep
-	As a result, conversion can lead to increase or decrease in energy level

 ## Solver
-	Solver can take from 0 timestep to a fixed timestep to move
-	For 0 timestep the solver would only look at short term gains
-	For fixed timestep the solver would look at long term gains as well
-	Solver’s goal is to maximize energy

## Maze-Master
-	Maze-master creates maze
-	It can be a fixed rule as well as GAN generated
-	For fixed rule, the maze and rooms would follow certain rules for treasure/trap
-	For GAN, the GAN would learn along with solver and create better maze
-	GAN would have certain restrictions to ensure it is not filled with traps all the time

# Application
## Games
-	It would create a game where the game AI tries to defeat the player
-	The game AI would learn from the player movements and would become better
## Speech
-	In speech each room would be individual sound (character for text)
-	Treasure would be reward for individual timestep based on user reaction/emotion
-	Energy would be overall reward based on the path taken
-	The maze can change based on user’s instantaneous feedback
-	The solver would try to find the proper answer
-	In this case the maze-master would work along with solver to ensure correct path is taken
-	The goal would be to get reward from user
-	The maze master would learn to make better maze based on user feedback, solver path and total reward
## Trading
-	It can be used to simulate trading
-	The maze would be the share market
-	The rooms would be each individual company
-	The treasure is the share of that company
-	The value of treasure is the share price
-	The timestep would be the real time
-	The time taken by individual to sell and receive money would be treasure-energy conversion timestep
-	The energy would be initial funds
-	The maze master would be the economy itself
-	The goal would be to maximize profit

