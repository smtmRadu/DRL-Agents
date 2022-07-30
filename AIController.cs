using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class AIController: Agent
{
    [Space, Header("Player Properties-------------------------------")]
    [SerializeField] private float maxSpeed = 5f;
    [SerializeField] private float speed = 5f;
    [SerializeField] private float jumpPower = 5f;
    Rigidbody2D rb;
    bool canJump = true;
    [SerializeField] Transform backTrap;


    protected override void Awake()
    {
        base.Awake();
        rb = GetComponent<Rigidbody2D>();
        backTrap = GameObject.Find("BackTrap").transform;
    }
    protected override void Update()
    {
        base.Update();

        if (behaviour == BehaviourType.Heuristic)
            Heuristic();
      
         
    }

    protected override void Heuristic()
    {
        if (Input.GetKey(KeyCode.A) && rb.velocity.magnitude < maxSpeed)
            rb.AddForce(Vector2.left * speed * Time.deltaTime * 10000);
        else if (Input.GetKey(KeyCode.D) && rb.velocity.magnitude < maxSpeed)
            rb.AddForce(Vector2.right * speed * Time.deltaTime * 10000);

        if (Input.GetKey(KeyCode.Space) && canJump)
        { 
            rb.AddForce(Vector2.up * jumpPower, ForceMode2D.Impulse);
            canJump = false;
        }
    }
    protected override bool OnOutputsReceived(float[] ActionBuffer)
    {
        if (!base.OnOutputsReceived(ActionBuffer))
            return false;

        if (ActionBuffer[0] >.5f && rb.velocity.magnitude < maxSpeed)
        {
            rb.AddForce(Vector2.right * speed * Time.deltaTime * 10000);
        }
        else if (rb.velocity.magnitude < maxSpeed)
        {
            rb.AddForce(Vector2.left * speed * Time.deltaTime * 10000);
        }
        if (ActionBuffer[0] >.5f && canJump)
        {
            
                rb.AddForce(Vector2.up * jumpPower, ForceMode2D.Impulse);
                canJump=false;
            
        }
        return true;
    }
    protected override void UpdateInputs(out float[] SensorBuffer)
    {
        
        base.UpdateInputs(out SensorBuffer);
        SensorBuffer[0] = transform.position.x;
        SensorBuffer[1] = transform.position.x - backTrap.position.x;

        return;
        string inputs = string.Join(" | ", SensorBuffer);
        Debug.Log("Inputs: " + inputs);
    }

    public void OnCollisionEnter2D(Collision2D collision)
    {
        if (collision.collider.name == "Platform")
            canJump = true;
        if(collision.collider.CompareTag("Enemy") && behaviour != BehaviourType.Heuristic && networkStatus)
        {
            SetReward(transform.position.x);
            EndEpisode();
        }
    }
}
