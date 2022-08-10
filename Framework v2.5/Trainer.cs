using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using MLFramework;
public class Trainer : TrainerBase
{
    protected override void Awake()
    {

    }
    protected override void Start()
    {

    }
    protected override void SetupTeam()
    {
        base.SetupTeam();
    }
    protected override void EnvironmentAction()
    {
        //Add actions for your environment, this method is called in Update 
        //Tip: use Time.deltaTime
    }
    protected override void OnEpisodeBegin()
    {

    }
    protected override void OnEpisodeEnd()
    {
        foreach (var ai in team)
        {
            if (ai.script.behaviour == BehaviourType.Learning)
            {
                //Add rewards to AI's that didn't ended their actions
            }
        }
    }
}
