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
        //Outputs are [0f,1f]
        //Trick to make them [-1f,1f] -> each float -= .5f then *= 2;
    }
    private void OnCollisionEnter2D(Collision2D collision)
    {
    }
}
