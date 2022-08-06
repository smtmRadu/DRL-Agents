using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Agent : AgentBase
{
    [Space, Header("===== AI Properties =====")]
    ///Create variables like speed, jumpPower, rotation etc.
    /// Also aditional variables to control components like RigidBody or SpriteRenderer
    /// EXAMPLE:
    public float speed = 1f;

    protected override void Awake()
    {
        base.Awake();
        //Awake or Set your variables if neccesary
    }
    protected override void Update()
    {
        base.Update();
        //Do whatever you want here
    }

    protected override void HeuristicTesting()
    {
        ///Implement Keyboard Input for your player.
        ///HeuristicTesting() is called in Update(), so use Time.deltaTime in your moving functions
        ///EXAMPLE: 
        /*if (Input.GetKey(KeyCode.A))
            transform.position += new Vector3(-1, 0, 0) * Time.deltaTime * speed;
        else if (Input.GetKey(KeyCode.D))
            transform.position += new Vector3(1, 0, 0) * Time.deltaTime * speed;

        if (Input.GetKey(KeyCode.W))
            transform.position += new Vector3(0, 1, 0) * Time.deltaTime * speed;
        else if (Input.GetKey(KeyCode.S))
            transform.position += new Vector3(0, -1, 0) * Time.deltaTime * speed;*/

    }
    protected override void UpdateInputs(out float[] SensorBuffer)
    {

        base.UpdateInputs(out SensorBuffer);//Sensor buffer is initialized here

        //SensorBuffer has a length of it's neural network inputs
        //Insert semnificative inputs for your AI if you want good results


        ///EXAMPLE: Turn ON ErrorPause in UnityConsole
        ///try
        ///{
        ///     SensorBuffer[0] = transform.position.x;
        ///     SensorBuffer[1] = transform.position.y;
        ///     SensorBuffer[2] = goal.transform.position.x;
        ///     SensorBuffer[3] = goal.transform.position.y;
        ///     SensorBuffer[4] = You can assign a value of 1 if a RaycastHit != null
        ///     SensorBuffer[5] = RaycastHit collision collider object ID (or some specific value) to know how to behave when he meets it
        ///     SensorBuffer[6] = level Index ( if you are planning to train your AI's for different Environments
        ///}
        ///catch(System.Exception exception) 
        ///{
        ///    Debug.LogError(exception);
        ///}
    }
    protected override bool OnOutputsReceived(in float[] ActionBuffer)
    {
        if (!base.OnOutputsReceived(in ActionBuffer))
            return false;

        //ActionBuffer has a length of it's neural outputs with values between [0f,1f]
        //You can use the outputs to make actions similar to HeuristicTesting
        //OnOuputsReveived() is called in Update(), so use Time.deltaTime in your moving functions
        ///EXAMPLE
       /* if (ActionBuffer[0] < .3f)
            transform.position += new Vector3(1, 0, 0) * Time.deltaTime * speed;
        else if(ActionBuffer[0] < .6f) transform.position += new Vector3(-1, 0, 0) * Time.deltaTime * speed;
        
        if (ActionBuffer[1] < .3f)
            transform.position += new Vector3(0, 1, 0) * Time.deltaTime * speed;
        else if (ActionBuffer[1] <.6f)
            transform.position += new Vector3(0, -1, 0) * Time.deltaTime * speed;*/


        //Do not modify this return
        return true;
    }

    public void OnCollisionEnter2D(Collision2D collision)
    {
        //Do other stuff here if you need

        ///EXAMPLE
        ///     if(collision.collider.CompareTag("Coin"))
        ///             AddReward(1f);
        ///     if(collision.collider.CompareTag("Trap"))
        ///             EndEpisode();
        ///     if(collision.collider.ComopareTag("FinnishLine")
        ///     {
        ///             SetReward(100f);
        ///             EndEpisode();
        ///     }        

    }
}
