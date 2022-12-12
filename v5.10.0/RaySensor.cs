using System;
using Unity.Burst.CompilerServices;
using Unity.VisualScripting;
using UnityEngine;

public class RaySensor : MonoBehaviour
{
    [HideInInspector] public float[] observations;
    [SerializeField, Tooltip("@scene world dimensions")] World world = World.World3D;
    [SerializeField, Tooltip("@LayerMask used when casting the rays")] LayerMask layerMask = ~0;
    [SerializeField, Range(1, 50), Tooltip("@size of the buffer equals the number of rays")] int rays = 5;
    [SerializeField, Range(1, 360)] int fieldOfView = 45;
    [SerializeField, Range(0, 359)] int rotationOffset = 0;
    [SerializeField, Range(1, 1000), Tooltip("@maximum length of the rays\n@when no collision, value of the observation is 0")] int distance = 30;
    [SerializeField, Range(0.01f, 10)] float sphereCastRadius = 0.5f;

    [Header("@Angles & Offsets")]
    [SerializeField, Range(-45, 45), Tooltip("@ray vertical tilt\n@not used in 2D world")] float rayTilt = 0;
    [SerializeField, Range(-5, 5), Tooltip("@ray vertical offset")] float xOffset = 0;
    [SerializeField, Range(-5, 5), Tooltip("@ray vertical offset")] float yOffset = 0;
    [SerializeField, Range(-5, 5), Tooltip("@ray vertical offset\n@not used in 2D world")] float zOffset = 0;

    [Header("@Gizmos Debug")]
    [SerializeField] Color rayColor = Color.green;
    [SerializeField] Color missRayColor = Color.red;

    private void Start()
    {
        CastRays();
    }
    private void Update()
    {
        CastRays();
    }
    private void OnDrawGizmos()
    {
        float oneAngle = rays == 1 ? 0 : (float)-fieldOfView / (rays - 1f);

        Vector3 startAngle;
        if (world == World.World3D)
            startAngle = Quaternion.Euler(rayTilt, (float)-oneAngle * (rays - 1) / 2 + rotationOffset, rayTilt) * transform.forward;
        else if (world == World.World2DforXY)
            startAngle = Quaternion.Euler(0, 0, (float)-oneAngle * (rays - 1) / 2 + rotationOffset) * this.transform.up;
        else //ZY
            startAngle = Quaternion.Euler((float)-oneAngle * (rays - 1) / 2 + rotationOffset, 0, 0) * transform.up;

        Vector3 castOrigin = transform.position + (transform.right * xOffset + transform.up * yOffset + transform.forward * zOffset) * transform.lossyScale.magnitude;

        float currentAngle = 0;

        for (int r = 0; r < rays; r++)
        {
            Vector3 rayDirection;
            if (world == World.World3D)
                rayDirection = Quaternion.Euler(0, currentAngle, 0) * startAngle;
            else if (world == World.World2DforXY)
                rayDirection = Quaternion.Euler(0, 0, currentAngle) * startAngle;
            else //ZY
                rayDirection = Quaternion.Euler(currentAngle, 0, 0) * startAngle;

            if (world == World.World3D)
            {
                RaycastHit hit;
                bool isHit = Physics.SphereCast(castOrigin, sphereCastRadius, rayDirection, out hit, distance, layerMask);
                if (isHit == true)
                {
                    Gizmos.color = rayColor;
                    Gizmos.DrawRay(castOrigin, rayDirection * hit.distance);
                    Gizmos.DrawWireSphere(castOrigin + rayDirection * hit.distance, sphereCastRadius);
                }
                else
                {
                    Gizmos.color = missRayColor;
                    Gizmos.DrawRay(castOrigin, rayDirection * distance);
                }
            }
            else
            {
                RaycastHit2D hit2D = Physics2D.CircleCast(castOrigin, sphereCastRadius, rayDirection, distance, layerMask);
                if (hit2D == true)
                {
                    Gizmos.color = rayColor;
                    Gizmos.DrawRay(castOrigin, rayDirection * hit2D.distance);
                    Gizmos.DrawWireSphere(castOrigin + rayDirection * hit2D.distance, sphereCastRadius);
                }
                else
                {
                    Gizmos.color = missRayColor;
                    Gizmos.DrawRay(castOrigin, rayDirection * distance);
                }

            }


            currentAngle += oneAngle;
        }


    }
    private void CastRays()
    {

        observations = new float[rays];
        float oneAngle = rays == 1 ? 0 : (float)-fieldOfView / (rays - 1f);

        Vector3 startAngle;
        if (world == World.World3D)
            startAngle = Quaternion.Euler(rayTilt, (float)-oneAngle * (rays - 1) / 2 + rotationOffset, rayTilt) * transform.forward;
        else if (world == World.World2DforXY)
            startAngle = Quaternion.Euler(0, 0, (float)-oneAngle * (rays - 1) / 2 + rotationOffset) * this.transform.up;
        else //ZY
            startAngle = Quaternion.Euler((float)-oneAngle * (rays - 1) / 2 + rotationOffset, 0, 0) * transform.up;

        Vector3 castOrigin = transform.position + (transform.right * xOffset + transform.up * yOffset + transform.forward * zOffset) * transform.lossyScale.magnitude;

        float currentAngle = 0;

        for (int r = 0; r < rays; r++)
        {
            Vector3 rayDirection;
            if (world == World.World3D)
                rayDirection = Quaternion.Euler(0, currentAngle, 0) * startAngle;
            else if (world == World.World2DforXY)
                rayDirection = Quaternion.Euler(0, 0, currentAngle) * startAngle;
            else //ZY
                rayDirection = Quaternion.Euler(currentAngle, 0, 0) * startAngle;

            if (world == World.World3D)
            {
                RaycastHit hit;
                bool isHit = Physics.SphereCast(castOrigin, sphereCastRadius, rayDirection, out hit, distance, layerMask);
                if (isHit == true)
                {
                    observations[r] = hit.distance;
                }
                else
                {
                    observations[r] = 0;
                }
            }
            else
            {
                RaycastHit2D hit2D = Physics2D.CircleCast(castOrigin, sphereCastRadius, rayDirection, distance, layerMask);
                if (hit2D == true)
                {
                    observations[r] = hit2D.distance;
                }
                else
                {
                    observations[r] = 0;
                }

            }


            currentAngle += oneAngle;
        }

    }

    public enum World
    {
        World3D,
        World2DforXY,
        World2DforZY,
    }
    //patch
    //v 1.1 updated and works properly
}
