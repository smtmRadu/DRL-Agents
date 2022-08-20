using UnityEngine;
using MLFramework;
public class Agent : AgentBase
{
    [Header("=== AI Properties ===")]
    public float speed;

    protected override void Update()
    {
        base.Update();
    }
    protected override void Manual()
    {
        //Implement a keyboard input for your AI - test only
    }
    protected override void CollectObservations(ref SensorBuffer sensorBuffer)
    {
        //sensorBuffer.Length == sensorSize
        //sensorBuffer.AddObservation(<observation>);
    }
    protected override void OnActionReceived(in ActionBuffer actionBuffer)
    {
        //actionBuffer.Length == actionSize 
        //actionBuffer.GetAction(<index>);
    }
    //Usefull Methods, use them in CollisionCollider2D, CollisionTrigger2D, Update(), etc. 
    //AddReward() - cumulative reward
    //SetReward() - static reward
    //EndAction() - behaviour becomes static
    //GetFitness() - current AI fitness

}
