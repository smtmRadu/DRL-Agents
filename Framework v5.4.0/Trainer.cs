using UnityEngine;
using MLFramework;
public class Trainer : TrainerBase
{
    protected override void Awake()
    {
        Application.runInBackground = true;
        base.Awake();
    }
    protected override void Start()
    {
        base.Start();
    }
    protected override void Update()
    {
        base.Update();
    }
    protected override void OnEpisodeBegin(ref GameObject environment)
    {
        //Actions after episode reset
        //Called for each environment separately
    }
    protected override void OnEpisodeEnd(ref AI ai)
    {
        // Actions before episode reset
        // Called for each agent separately
            // ai.agent - used to access the agent gameobject
            // ai.script - used to acces Agent Script 
            // ai.fitness - used to get it's current fitness
            // To add reward even if agent's action ended, use ai.script.AddFitness(value,true) [reward will be added even if AI behaviour becomes Static]
    }
}
