# MLAgent
Set of C# files that can be used to train the AI in Unity Engine.

The Neural Network Class contains the code for the implementation of a ANN that uses EvolutionaryTraining -> random Mutation per Weight (probably in the future i will make an option for backpropagation).
The Agent Class and TrainerAgent Class are the only one that are used in the project following the next steps:

1.The AI Controller Script must Inherit the Agent class (Agent class already inerits from MonoBehaviour)
2.In the Inspector the variables from the Agent Class are placed firstly on the Script Component
3.In the AI Controller Script, some virtual methods must be overrided to provide training
4.The description must be completed in the future...
