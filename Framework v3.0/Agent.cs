using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using MLFramework;
public class Agent : AgentBase
{
    protected override void Update()
    {
        base.Update();
    }
    protected override void Manual()
    {
        //Implement a keyboard input for your AI - test only
    }
    protected override void CollectObservations(ref float[] SensorBuffer)
    {
        //Fullfill SensorBuffer with observations
    }
    protected override void OnActionReceived(in float[] ActionBuffer)
    {
        //ActionBuffer outputs are in [-1f,1f] range
    }
    private void OnCollisionEnter2D(Collision2D collision)
    {
        //Usefull Methods:
            //AddReward()
            //SetReward()
            //EndAction()
            //GetReward()
    }
}
