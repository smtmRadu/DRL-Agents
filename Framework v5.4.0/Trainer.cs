using UnityEngine;
using MLFramework;
public class Trainer : TrainerBase
{
    [Space, Header("===== Other =====")]
    public GameObject goal;

    protected override void Awake()
    {
        Application.runInBackground = true;
        base.Awake();
    }
    protected override void Start()
    {
        base.Start();
    }
    protected override void SetupTeam()
    {
        base.SetupTeam();
        //Additional changes to the AI's before starting the training session
    }
    protected override void EnvironmentAction()
    {
        //Add actions for your environment
    }
    protected override void OnEpisodeBegin(ref GameObject[] Environments)
    {
        //Actions after episode reset
        //Parse the parameter to modify the "same" object in each environment
    }
    protected override void OnEpisodeEnd(ref AI ai)
    {
        // Actions before episode reset
        // Parse the parameter to modify all AI's if needed
        // ai.agent - used to access the agent gameobject
        // ai.script - used to acces Agent Script 
        // ai.fitness - used to get it's current fitness
        // To add reward even if Ai's action ended, use ai.script.AddFitness(value,true) [reward will be added even if AI behaviour becomes Static]
    }
}
