using System;
using Unity.VisualScripting;
using UnityEngine;
using UnityEditor;

namespace DRLAgents
{
    [AddComponentMenu("DRL Agents/Ray Sensor")]
    public class RaySensor : MonoBehaviour
    {

        [HideInInspector] public float[] observations;
        [SerializeField, Tooltip("@scene type")] World world = World.World3d;
        [SerializeField, Tooltip("@LayerMask used when casting the rays")] LayerMask layerMask = ~0;
        [SerializeField, Tooltip("@observation value returned by the rays")] Info info = Info.Distance;
        [SerializeField, Range(1, 50), Tooltip("@size of the buffer equals the number of rays")] int rays = 5;
        [SerializeField, Range(1, 360)] int fieldOfView = 45;
        [SerializeField, Range(0, 359)] int rotationOffset = 0;
        [SerializeField, Range(1, 1000), Tooltip("@maximum length of the rays\n@when no collision, value of the observation is 0")] int distance = 30;
        [SerializeField, Range(0.01f, 10)] float sphereCastRadius = 0.5f;

        [Space(10)]
        [SerializeField, Range(-45, 45), Tooltip("@ray vertical tilt\n@not used in 2D world")] float tilt = 0;
        [SerializeField, Range(-5, 5), Tooltip("@ray X axis offset")] float xOffset = 0;
        [SerializeField, Range(-5, 5), Tooltip("@ray Y axis offset")] float yOffset = 0;
        [SerializeField, Range(-5, 5), Tooltip("@ray Z axis offset\n@not used in 2D world")] float zOffset = 0;

        [Space(10)]
        [SerializeField] Color rayColor = Color.green;
        [SerializeField] Color missRayColor = Color.red;

        Agent agent;
        bool simStarted = false;
        private void Awake()
        {
            simStarted = true;
        }
        private void Start()
        {
            agent = GetComponent<Agent>();
            CastRays();
        }
        private void Update()
        {
            CastRays();
        }
        private void OnDrawGizmos()
        {
            if (simStarted && agent.behavior == DRLAgents.BehaviorType.Static)
                return;

            float oneAngle = rays == 1 ? 0 : (float)-fieldOfView / (rays - 1f);

            float begin = (float)-oneAngle * (rays - 1) / 2 + rotationOffset;
            Vector3 startAngle;

            if (world == World.World3d)
                startAngle = Quaternion.AngleAxis(tilt, transform.right) * Quaternion.AngleAxis(tilt, transform.forward) * Quaternion.AngleAxis(begin, transform.up) * transform.forward;
            else //world2d
                startAngle = Quaternion.AngleAxis(begin, transform.forward) * transform.up;

            Vector3 castOrigin = transform.position + (transform.right * xOffset + transform.up * yOffset + transform.forward * zOffset) * transform.lossyScale.magnitude;

            float currentAngle = 0;

            for (int r = 0; r < rays; r++)
            {
                Vector3 rayDirection;
                if (world == World.World3d) //3d
                {
                    rayDirection = Quaternion.AngleAxis(currentAngle, transform.up) * startAngle;

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
                else //2d
                {
                    rayDirection = Quaternion.AngleAxis(currentAngle, transform.forward) * startAngle;
                    
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
            if (agent.behavior == BehaviorType.Static)
                return;

            observations = new float[rays];
            float oneAngle = rays == 1 ? 0 : (float)-fieldOfView / (rays - 1f);

            float begin = (float)-oneAngle * (rays - 1) / 2 + rotationOffset;
            Vector3 startAngle;
            if (world == World.World3d)
                startAngle = Quaternion.AngleAxis(tilt, transform.right) * Quaternion.AngleAxis(tilt, transform.forward) * Quaternion.AngleAxis(begin, transform.up) * transform.forward;
            else //world2d
                startAngle = Quaternion.AngleAxis(begin, transform.forward) * transform.up;


            Vector3 castOrigin = transform.position + (transform.right * xOffset + transform.up * yOffset + transform.forward * zOffset) * transform.lossyScale.magnitude;

            float currentAngle = 0;

            for (int r = 0; r < rays; r++)
            {
                
                if (world == World.World3d)
                {
                    Vector3 rayDirection = Quaternion.AngleAxis(currentAngle, transform.up) * startAngle;
                    CastRay3D(r, castOrigin, sphereCastRadius, rayDirection, distance, layerMask);
                }
                else
                {
                    Vector3 rayDirection = Quaternion.AngleAxis(currentAngle, transform.forward) * startAngle;
                    CastRay2D(r, castOrigin, sphereCastRadius, rayDirection, distance, layerMask);
                }
              
                currentAngle += oneAngle;
            }

        }
        private void CastRay3D(int index, Vector3 castOrigin, float sphereCastRadius, Vector3 rayDirection, float distance, LayerMask layerMask)
        {
            RaycastHit hit;
            bool isHit = Physics.SphereCast(castOrigin, sphereCastRadius, rayDirection, out hit, distance, layerMask);
            if (isHit == true)
            {
                if (info == Info.Distance)
                    observations[index] = hit.distance;
                else
                {
                    RayDetectable inf = hit.transform.GetComponent<RayDetectable>();
                    observations[index] = inf == null ? 0 : inf.infoValue;
                }
            }
            else
            {
                observations[index] = 0;
            }
        }
        private void CastRay2D(int index, Vector3 castOrigin, float sphereCastRadius, Vector3 rayDirection, float distance, LayerMask layerMask)
        {
            RaycastHit2D hit = Physics2D.CircleCast(castOrigin, sphereCastRadius, rayDirection, distance, layerMask);
            if (hit == true)
            {
                if (info == Info.Distance)
                    observations[index] = hit.distance;
                else
                {
                    RayDetectable inf = hit.transform.GetComponent<RayDetectable>();
                    observations[index] = inf == null ? 0 : inf.infoValue;
                }
            }
            else
            {
                observations[index] = 0;
            }
        }

        public enum World
        {
            [Tooltip("@horizontal plane cast")]
            World3d,
            [Tooltip("@vertical plane cast\n@can be used for 3d worlds too")]
            World2d,
        }
        public enum Info
        {
            [Tooltip("@distance to the hit object")]
            Distance,
            [Tooltip("@[Ray Detectable] hit object value")]
            Value,
        }
        //patch
        //v 2.0 updated and works properly
    }
    [CustomEditor(typeof(RaySensor)), CanEditMultipleObjects]
    class ScriptlessEditor : Editor
    {
        private static readonly string[] _dontIncludeMe = new string[] { "m_Script" };

        public override void OnInspectorGUI()
        {
            serializedObject.Update();

            DrawPropertiesExcluding(serializedObject, _dontIncludeMe);

            serializedObject.ApplyModifiedProperties();
        }
    }
}