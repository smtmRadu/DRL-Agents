using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using MLFramework;
public class Agent : AgentBase
{

    private void Awake()
    {

    }
    protected override void Update()
    {
        base.Update();
    }
    protected override void Manual()
    {

    }
    protected override void CollectObservations(ref float[] SensorBuffer)
    {

    }
    protected override void OnActionReceived(in float[] ActionBuffer)
    {
        //ActionBuffer outputs are in [-1f,1f] range
    }
    private void OnCollisionEnter2D(Collision2D collision)
    {
        //AddReward()
        //EndAction()
        //ResetEpisode()
    }
}
