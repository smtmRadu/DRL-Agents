using System.Collections;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using UnityEditor;
using UnityEngine;

namespace DRLAgents
{
    [AddComponentMenu("DRL Agents/Ray Detectable")]
    public class RayDetectable : MonoBehaviour
    {
        [Tooltip("@value returned by a ray hitting this object\n" +
            "@objects that doesn't have this script return value 0, so do not set this variable to 0\n" +
            "@recommended to use negative values too")] public float infoValue = 1;
       
    }

    [CustomEditor(typeof(RayDetectable), true), CanEditMultipleObjects]
    class ScriptlessRayInfo : Editor
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