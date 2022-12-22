using System.Collections;
using System.Collections.Generic;
using System.Text;
using UnityEditor;
using UnityEngine;
using System;
using System.Linq;
using System.Linq.Expressions;

namespace DRLAgents
{
    //-ReadOnly attribute-----------------------------
    public class ReadOnlyAttribute : PropertyAttribute
    {

    }

    [CustomPropertyDrawer(typeof(ReadOnlyAttribute))]
    public class ReadOnlyDrawer : PropertyDrawer
    {
        public override float GetPropertyHeight(SerializedProperty property,
                                                GUIContent label)
        {
            return EditorGUI.GetPropertyHeight(property, label, true);
        }

        public override void OnGUI(Rect position,
                                   SerializedProperty property,
                                   GUIContent label)
        {
            GUI.enabled = false;
            EditorGUI.PropertyField(position, property, label, true);
            GUI.enabled = true;
        }
    }

    //-------------------------------------------------
    internal readonly struct Functions
    {
        internal readonly struct Activation
        {
            static public float ActivationFunctionBinaryStep(float value)
            {
                if (value < 0)
                    return 0;
                else return 1;
            }
            static public float ActivationFunctionSigmoid(float value)
            {
                //values range [0,1]
                // Function is x = 1/(1 + e^(-x))
                return (float)1f / (1f + Mathf.Exp(-value));
            }
            static public float ActivationFunctionTanh(float value)
            {
                return (float)System.Math.Tanh((double)value);
                /*
                 //Other variant is to shift the sigmoid function
                   return (float)2f / (1f + Mathf.Exp(-2*value)) - 1;


                 */
            }
            static public float ActivationFunctionReLU(float value)
            {
                return Mathf.Max(0, value);
            }
            static public float ActivationFunctionLeakyReLU(float value, float alpha = 0.2f)
            {
                if (value > 0)
                    return value;
                else return value * alpha;
            }
            static public float ActivationFunctionSiLU(float value)
            {
                return value * ActivationFunctionSigmoid(value);
            }
            static public void ActivationFunctionSoftMax(ref float[] values)
            {
                float sum = 0f;
                for (int i = 0; i < values.Length; i++)
                {
                    values[i] = Mathf.Exp(values[i]);
                    sum += values[i];
                }
                for (int i = 0; i < values.Length; i++)
                {
                    values[i] /= sum;
                }
            }
        }
        internal readonly struct Derivatives
        {
            static public float DerivativeTanh(float value)
            {
                return 1f - (float)Math.Pow(Math.Tanh(value), 2);
            }
            static public float DerivativeSigmoid(float value)
            {
                return Activation.ActivationFunctionSigmoid(value) * (1 - Activation.ActivationFunctionSigmoid(value));
            }
            static public float DerivativeBinaryStep(float value)
            {
                return 0;
            }
            static public float DerivativeReLU(float value)
            {
                if (value < 0)
                    return 0;
                else return 1;
            }
            static public float DerivativeLeakyReLU(float value, float alpha = 0.2f)
            {
                if (value < 0)
                    return alpha;
                else return 1;
            }
            static public float DerivativeSiLU(float value)
            {
                return (1 + Mathf.Exp(-value) + value * Mathf.Exp(-value)) / Mathf.Pow((1 + Mathf.Exp(-value)), 2);
                //return ActivationFunctionSigmoid(value) * (1 + value * (1 - ActivationFunctionSigmoid(value))); -> works the same
            }
            static public void DerivativeSoftMax(ref float[] values)
            {
                float sum = 0f;

                foreach (float item in values)
                    sum += Mathf.Exp(item);


                for (int i = 0; i < values.Length; i++)
                {
                    float ePowI = Mathf.Exp(values[i]);
                    values[i] = (ePowI * sum - ePowI * ePowI) / (sum * sum);
                }
            }
        }
        internal readonly struct Cost
        {
            static public float Quadratic(float outputActivation, float expectedOutput)
            {
                float error = outputActivation - expectedOutput;
                return error * error;
            }
            static public float QuadraticDerivative(float outputActivation, float expectedOutput)
            {
                return 2 * (outputActivation - expectedOutput);
            }
            static public float Absolute(float outputActivation, float expectedOutput)
            {
                return Mathf.Abs(outputActivation - expectedOutput);
            }
            static public float AbsoluteDerivative(float outputActivation, float expectedOutput)
            {
                if ((outputActivation - expectedOutput) > 0)
                    return 1;
                else return -1;
            }
            static public float CrossEntropy(float outputActivation, float expectedOutput)
            {
                double v = (expectedOutput == 1) ? -System.Math.Log(outputActivation) : -System.Math.Log(1 - outputActivation);
                return double.IsNaN(v) ? 0 : (float)v;
            }
            static public float CrossEntropyDerivative(float outputActivation, float expectedOutput)
            {
                if (outputActivation == 0 || outputActivation == 1)
                    return 0;
                return (-outputActivation + expectedOutput) / (outputActivation * (outputActivation - 1));
            }
        }
        internal readonly struct Mutation
        {
            static public void ClassicMutation(ref float weight)
            {
                float randNum = UnityEngine.Random.Range(0f, 10f);

                if (randNum <= 2f)//20% chance of flip sign of the weightOrBias
                {
                    weight *= -1f;
                }
                else if (randNum <= 4f)//20% chance of fully randomize weightOrBias
                {
                    weight = UnityEngine.Random.Range(-.5f, .5f);
                }
                else if (randNum <= 6f)//20% chance of increase to 100 - 200 %
                {
                    float factor = UnityEngine.Random.value + 1f;
                    weight *= factor;
                }
                else if (randNum <= 8f)//20% chance of decrease in range 0 - 100 %
                {
                    float factor = UnityEngine.Random.value;
                    weight *= factor;
                }
                else
                {
                }//20% chance of NO MUTATION

            }
            static public void LightPercentageMutation(ref float weight)
            {
                //increase/decrease all to a max of 50%
                float sign = UnityEngine.Random.value;
                float factor;
                if (sign > .5f)
                {
                    factor = UnityEngine.Random.Range(1f, 1.5f);
                }
                else
                {
                    factor = UnityEngine.Random.Range(.5f, 1f);
                }
                weight *= factor;
            }
            static public void StrongPercentagegMutation(ref float weight)
            {
                //increase/decrease all to a max of 100%

                float sign = UnityEngine.Random.value;
                float factor;
                if (sign > .5f)//increase
                {
                    factor = UnityEngine.Random.value + 1f;
                }
                else//decrease
                {
                    factor = UnityEngine.Random.value;

                }
                weight *= factor;

            }
            static public void LightValueMutation(ref float weight)
            {
                // + 0 -> .5f or  - 0 -> .5f
                float randNum = UnityEngine.Random.Range(-.5f, .5f);
                weight += randNum;
            }
            static public void StrongValueMutation(ref float weight)
            {
                float randNum = UnityEngine.Random.Range(-1f, 1f);
                weight += randNum;
            }
            static public void ChaoticMutation(ref float weight)
            {
                float chance = UnityEngine.Random.value;
                if (chance < .125f)
                    weight = Functions.Initialization.RandomValueInCustomDeviationDistribution(0.15915f, 2f, 0.3373f);
                else if (chance < .3f)
                    ClassicMutation(ref weight);
                else if (chance < .475f)
                    LightPercentageMutation(ref weight);
                else if (chance < .65f)
                    StrongPercentagegMutation(ref weight);
                else if (chance < .825f)
                    LightValueMutation(ref weight);
                else
                    StrongValueMutation(ref weight);

            }
        }
        internal readonly struct Initialization
        {
            /// <summary>
            /// Return a random value [-1,1] != 0
            /// </summary>
            /// <returns></returns>
            static public float RandomValue()
            {
                if (UnityEngine.Random.value > 0.5f)
                    return UnityEngine.Random.value;
                else
                    return -UnityEngine.Random.value;
            }
            static public float RandomValueInCustomDeviationDistribution(float l, float k, float z)
            {
                float x = UnityEngine.Random.value;
                float sign = UnityEngine.Random.value;
                if (sign > .5f)
                    return (float)Mathf.Pow(-Mathf.Log(2f * l * Mathf.PI * Mathf.Pow(x, 2f)) * z, 1f / k);
                else
                    return (float)-Mathf.Pow(-Mathf.Log(2f * l * Mathf.PI * Mathf.Pow(x, 2f)) * z, 1f / k);


            }
            static public float RandomInNormalDistribution(System.Random rng, float mean, float standardDeviation)
            {
                float x1 = (float)(1 - rng.NextDouble());
                float x2 = (float)(1 - rng.NextDouble());

                float y1 = Mathf.Sqrt(-2.0f * Mathf.Log(x1)) * Mathf.Cos(2.0f * (float)Math.PI * x2);
                return y1 * standardDeviation + mean;
            }

        }
        internal readonly struct ArrayConversion
        {
            static public void ConvertStrArrToIntArr(string[] str, ref int[] arr)
            {
                for (int i = 0; i < arr.Length; i++)
                {
                    try
                    {
                        arr[i] = int.Parse(str[i]);
                    }
                    catch (System.Exception e)
                    {
                        Debug.LogError(str[i] + " : " + e);
                    }

                }
            }
            static public void ConvertStrArrToFloatArr(string[] str, ref float[] arr)
            {
                for (int i = 0; i < arr.Length; i++)
                {
                    try
                    {
                        arr[i] = float.Parse(str[i]);
                    }
                    catch (System.Exception e)
                    {
                        Debug.LogError(str[i] + " : " + e);
                    }

                }
            }
        }

    }
    internal readonly struct ColorConvertor
    {
        public static void ConvertColorToColor32(Color color, ref Color32 color32)
        {
            color32.r = System.Convert.ToByte(color.r * 255f);
            color32.g = System.Convert.ToByte(color.g * 255f);
            color32.b = System.Convert.ToByte(color.b * 255f);
            color32.a = System.Convert.ToByte(color.a * 255f);
        }
        public static string GetRichTextColorFromColor32(Color32 color)
        {
            string clr = "#";
            clr += GetHexFrom(color.r);
            clr += GetHexFrom(color.g);
            clr += GetHexFrom(color.b);
            return clr;
        }
        public static string GetHexFrom(int value)
        {
            ///The format of the Number is returned in XX Format
            int firstValue = value;

            StringBuilder hexCode = new StringBuilder();
            int remainder;

            while (value > 0)
            {
                remainder = value % 16;
                value -= remainder;
                value /= 16;

                hexCode.Append(GetHexDigFromIntDig(remainder));
            }
            if (firstValue <= 15)
                hexCode.Append("0");
            if (firstValue == 0)//Case 0, we need to return 00
                hexCode.Append("0");

            string hex = hexCode.ToString();
            ReverseString(ref hex);
            return hex;
        }
        public static string GetHexDigFromIntDig(int value)
        {
            if (value < 0 || value > 15)
            {
                Debug.LogError("Value Parsed is not a Digit in HexaDecimal");
                return null;
            }
            if (value < 10)
                return value.ToString();
            else if (value == 10)
                return "A";
            else if (value == 11)
                return "B";
            else if (value == 12)
                return "C";
            else if (value == 13)
                return "D";
            else if (value == 14)
                return "E";
            else if (value == 15)
                return "F";
            else return null;
        }
        public static void ReverseString(ref string str)
        {
            char[] charArray = str.ToCharArray();
            System.Array.Reverse(charArray);
            str = new string(charArray);
        }
    }
    internal readonly struct Crossover
    {
        public static void InitOffspring(NeuralNetwork offspring, NeuralNetwork parent1, NeuralNetwork parent2, CrossoverType type, float p1Fitness, float p2Fitness)
        {
            switch(type)
            {
                case CrossoverType.Average:
                    AverageCross(offspring, parent1, parent2);
                    break;
                case CrossoverType.Weighted:
                    WeightedCross(offspring, parent1, parent2, p1Fitness, p2Fitness);
                    break;
                case CrossoverType.Uniform:
                    UniformCross(offspring, parent1, parent2);
                    break;
                case CrossoverType.OnePoint:
                    OnePointCross(offspring, parent1, parent2);
                    break;
                default:
                    Debug.LogError("Crossover type not set!");
                    break;
            }
        }
        private static void AverageCross(NeuralNetwork offspring, NeuralNetwork p1, NeuralNetwork p2)
        {
            int[] format = offspring.GetLayers();
            for (int l = 1; l < format.Length; l++)
            {
                for (int n = 0; n < format[l]; n++)
                {
                    for (int p = 0; p < format[l - 1]; p++)
                    {
                        float newWval = (p1.GetWeight(l - 1, n, p) + p2.GetWeight(l - 1, n, p)) / 2;
                        offspring.SetWeight(l - 1, n, p, newWval);
                    }

                    float newBval = (p1.GetBias(l, n) + p2.GetBias(l, n)) / 2;
                    offspring.SetBias(l, n, newBval);
                }
            }
        }
        private static void OnePointCross(NeuralNetwork offspring, NeuralNetwork p1, NeuralNetwork p2)
        {
            int[] format = offspring.GetLayers();
            int center = UnityEngine.Random.Range(1, format.Sum());
            int currentNeuron = 0;

            for (int l = 1; l < format.Length; l++)
            {
                for (int n = 0; n < format[l]; n++)
                {
                    currentNeuron++;
                    for (int p = 0; p < format[l - 1]; p++)
                    {
                        float newWval = currentNeuron <= center ? p1.GetWeight(l - 1, n, p) : p2.GetWeight(l - 1, n, p);
                        offspring.SetWeight(l - 1, n, p, newWval);
                    }

                    float newBval = currentNeuron <= center ? p1.GetBias(l, n) : p2.GetBias(l, n);
                    offspring.SetBias(l, n, newBval);
                }
            }
        }
        private static void UniformCross(NeuralNetwork offspring, NeuralNetwork p1, NeuralNetwork p2)
        {
            int[] format = offspring.GetLayers();
            for (int l = 1; l < format.Length; l++)
            {
                for (int n = 0; n < format[l]; n++)
                {
                    for (int p = 0; p < format[l - 1]; p++)
                    {
                        float newWval = UnityEngine.Random.value < 0.5f ? p1.GetWeight(l - 1, n, p) : p2.GetWeight(l - 1, n, p);
                        offspring.SetWeight(l - 1, n, p, newWval);
                    }

                    float newBval = UnityEngine.Random.value < 0.5f ? p1.GetBias(l, n) : p2.GetBias(l, n);
                    offspring.SetBias(l, n, newBval);
                }
            }
        }
        private static void WeightedCross(NeuralNetwork offspring, NeuralNetwork p1, NeuralNetwork p2, float? fit1, float? fit2)
        {
            float sum = (float)fit1 + (float)fit2;

            float firstParentWeight = (float)fit1 / sum;
            float secondParentWeight = (float)fit2 / sum;

            

            int[] format = offspring.GetLayers();
            for (int l = 1; l < format.Length; l++)
            {
                for (int n = 0; n < format[l]; n++)
                {
                    for (int p = 0; p < format[l - 1]; p++)
                    {
                        float newWval = p1.GetWeight(l - 1, n, p) * firstParentWeight + p2.GetWeight(l - 1, n, p) * secondParentWeight;
                        offspring.SetWeight(l - 1, n, p, newWval);
                    }

                    float newBval = p1.GetBias(l, n) *firstParentWeight + p2.GetBias(l, n) * secondParentWeight;
                    offspring.SetBias(l, n, newBval);
                }
            }
        }
    }

    public struct AI : ICloneable
    {
        /// <summary>
        /// Agent gameobject.
        /// </summary>
        public GameObject agent;
        /// <summary>
        /// Agent script component of your agent.
        /// </summary>
        public Agent script;
        /// <summary>
        /// Current agent fitness of your agent.
        /// </summary>
        public float fitness;


        public object Clone()
        {
            return new AI
            {
                agent = null,
                script = this.script,
                fitness = this.fitness
            };
        }
    }
    public struct PosAndRot
    {
        public Vector3 position, scale;
        public Quaternion rotation;
        public PosAndRot(Vector3 pos, Vector3 scl, Quaternion rot)
        {
            position = pos;
            scale = scl;
            rotation = rot;
        }
        public PosAndRot(UnityEngine.Transform transform)
        {
            position = transform.position;
            scale = transform.localScale;
            rotation = transform.rotation;
        }
    }
    public struct SensorBuffer
    {
        private float[] buffer;
        private int sizeIndex;
        public SensorBuffer(int capacity)
        {
            buffer = new float[capacity];
            for (int i = 0; i < capacity; i++)
                buffer[i] = 0;
            sizeIndex = 0;
        }
        /// <summary>
        /// Returns the array that contains all the input values .
        /// </summary>
        /// <returns>float[] with all values</returns>
        public float[] GetBuffer()
        {
            return buffer;
        }
        public int GetBufferCapacity()
        {
            if (buffer == null)
                return 0;
            else return buffer.Length;
        }


        /// <summary>
        /// Appends a float value to the SensorBuffer.
        /// </summary>
        /// <param name="observation1">Value of the observation</param>
        public void AddObservation(float observation1)
        {
            if (sizeIndex == buffer.Length)
            {
                Debug.Log("SensorBuffer is full. Increase the space size or remove this observation.");
                return;
            }
            buffer[sizeIndex++] = observation1;
        }
        /// <summary>
        ///  Appends an int value to the SensorBuffer.
        /// </summary>
        /// <param name="observation1">Value of the observation</param>
        public void AddObservation(int observation1)
        {
            if (sizeIndex == buffer.Length)
            {
                Debug.Log("SensorBuffer is full. Increase the space size or remove this observation.");
                return;
            }
            buffer[sizeIndex++] = observation1;
        }
        /// <summary>
        /// Appends an unsigned int value to the SensorBuffer.
        /// </summary>
        /// <param name="observation1">Value of the observation</param>
        public void AddObservation(uint observation1)
        {
            if (sizeIndex == buffer.Length)
            {
                Debug.Log("SensorBuffer is full. Increase the space size or remove this observation.");
                return;
            }
            buffer[sizeIndex++] = observation1;
        }
        /// <summary>
        /// Appends a Vector2 value to the SensorBuffer.
        /// </summary>
        /// <param name="observation2">Value of the observation</param>
        public void AddObservation(Vector2 observation2)
        {
            if (buffer.Length - sizeIndex < 2)
            {
                Debug.Log("SensorBuffer available space is " + (buffer.Length - sizeIndex) + ". Vector2 observation of size 2 is too large.");
                return;
            }
            buffer[sizeIndex++] = observation2.x;
            buffer[sizeIndex++] = observation2.y;
        }
        /// <summary>
        /// Appends a Vector3 value to the SensorBuffer.
        /// </summary>
        /// <param name="observation3">Value of the observation</param>
        public void AddObservation(Vector3 observation3)
        {

            if (buffer.Length - sizeIndex < 3)
            {
                Debug.Log("SensorBuffer available space is " + (buffer.Length - sizeIndex) + ". Vector3 observation of size 3 is too large.");
                return;
            }
            buffer[sizeIndex++] = observation3.x;
            buffer[sizeIndex++] = observation3.y;
            buffer[sizeIndex++] = observation3.z;
        }
        /// <summary>
        /// Appends a Vector4 value to the SensorBuffer.
        /// </summary>
        /// <param name="observation4">Value of the observation</param>
        public void AddObservation(Vector4 observation4)
        {

            if (buffer.Length - sizeIndex < 4)
            {
                Debug.Log("SensorBuffer available space is " + (buffer.Length - sizeIndex) + ". Vector4 observation of size 4 is too large.");
                return;
            }

            buffer[sizeIndex++] = observation4.x;
            buffer[sizeIndex++] = observation4.y;
            buffer[sizeIndex++] = observation4.z;
            buffer[sizeIndex++] = observation4.w;
        }
        /// <summary>
        /// Appends a Quaternion value to the SensorBuffer.
        /// </summary>
        /// <param name="observation4">Value of the observation</param>
        public void AddObservation(Quaternion observation4)
        {
            if (buffer.Length - sizeIndex < 4)
            {
                Debug.Log("SensorBuffer available space is " + (buffer.Length - sizeIndex) + ". Quaternion observation of size 4 is too large.");
                return;
            }
            buffer[sizeIndex++] = observation4.x;
            buffer[sizeIndex++] = observation4.y;
            buffer[sizeIndex++] = observation4.z;
            buffer[sizeIndex++] = observation4.w;
        }
        /// <summary>
        /// Appends a Transform value to the SensorBuffer.
        /// </summary>
        /// <param name="observation10">Value of the observation</param>
        public void AddObservation(UnityEngine.Transform obsevation10)
        {
            if (buffer.Length - sizeIndex < 10)
            {
                Debug.Log("SensorBuffer available space is " + (buffer.Length - sizeIndex) + ". Transform observation of size 10 is too large.");
                return;
            }
            AddObservation(obsevation10.position);
            AddObservation(obsevation10.localScale);
            AddObservation(obsevation10.rotation);
        }
        /// <summary>
        /// Appends an array of float values to the SensorBuffer.
        /// </summary>
        /// <param name="observations">Values of the observations</param>
        public void AddObservation(float[] observations)
        {
            if (buffer.Length - sizeIndex < observations.Length)
            {
                Debug.Log("SensorBuffer available space is " + (buffer.Length - sizeIndex) + ". Float array observations is too large.");
                return;
            }
            foreach (var item in observations)
            {
                AddObservation(item);
            }
        }
        /// <summary>
        /// Appends the distances of each RayCast by the RaySensor to SensorBuffer.
        /// </summary>
        /// <param name="raySensor">RaySensor object</param>
        public void AddObservation(RaySensor raySensor)
        {
            if (raySensor == null)
            {
                Debug.LogError("<color=red>RaySensor added as an observation is null!.</color>");
                return;
            }
            AddObservation(raySensor.observations);
        }
    }
    public struct ActionBuffer
    {
        private float[] buffer;
        public ActionBuffer(float[] actions)
        {
            buffer = actions;
        }
        public ActionBuffer(int capacity)
        {
            buffer = new float[capacity];
        }

        /// <summary>
        /// Get the buffer array with every action values.
        /// <para>Can be used instead of using GetAction() method.</para>
        /// </summary>
        /// <returns>float[] copy of the buffer</returns>
        public float[] GetBuffer()
        {
            return buffer;
        }
        /// <summary>
        /// 
        /// </summary>
        /// <returns>Total actions number</returns>
        public int GetBufferCapacity()
        {
            return buffer != null ? buffer.Length : 0;
        }
        /// <summary>
        /// Returns the value from the index parameter.
        /// </summary>
        /// <param name="index">The index of the action from ActionBuffer.</param>
        /// <returns>float</returns>
        public float GetAction(uint index)
        {
            try
            {
                return buffer[index];
            }
            catch { Debug.LogError("Action index out of range."); }
            return 0;
        }
        /// <summary>
        /// Sets the action from ActionBuffer with a specific value.
        /// </summary>
        /// <param name="index">The index of the action from ActionBuffer</param>
        /// <param name="action1">The value of the action to be set</param>
        public void SetAction(uint index, float action1)
        {
            buffer[index] = action1;
        }
        /// <summary>
        /// Returns the index of the max value from ActionBuffer.
        /// <para>Usually used when SoftMax is the output activation function.</para>
        /// </summary>
        /// <returns>The index or -1 if all elements are equal.</returns>
        public int GetIndexOfMaxValue()
        {
            float max = float.MinValue;
            int index = -1;
            bool equal = true;
            for (int i = 0; i < buffer.Length; i++)
            {
                if (i > 0 && buffer[i] != buffer[i - 1])
                    equal = false;

                if (buffer[i] > max)
                {
                    max = buffer[i];
                    index = i;
                }
            }
            return equal == true ? -1 : index;

        }
    }
    internal struct Node
    {
        public float valueIn,//before activation
                     valueOut,//after activation
                     costValue;
    }
    internal struct Sample
    {
        public float[] inputs;
        public float[] expectedOutputs;
    }
   
    public enum InteractionType
    {
        [Tooltip("@static environment\n@single environment\n@multiple agents\n@agent model is used as a starting position")]
        NotSpecified,

        //Agents overlap eachother, environmental objects are common
        [Tooltip("@agents are overlapping in the same environment(s)\n@if no starting model inside the environment, agent model is used as a starting position")]
        MoreAgentsPerEnvironment,

        //Agents train separately, environmental objects are personal for each agent
        [Tooltip("@one agent per each environment found\n@usually used to let just 1 agent interact with the environment")]
        OneAgentPerEnvironment,
    }
    public enum BehaviorType
    {
        [Tooltip("Doesn't move")]
        Static,
        [Tooltip("Can move only by user input\n@override Heuristic()\n@override OnActionReceived()")]
        Manual,
        [Tooltip("Moves independently\n@override CollectObservations()\n@override OnActionReceived()")]
        Self,
        [Tooltip("Trains by user input\nNo Trainer required\n@override CollectObservations()\n@override Heuristic()\n@override OnActionReceived()")]
        Heuristic,


    }

    public enum MutationType
    {
        [Tooltip("20% -> * (-1) " +
            "\n20% -> +.5f | -.5f" +
            "\n20% -> + 0%~100%" +
            "\n20% -> - 0%~100%" +
            "\n20% -> no mutation")]
        Classic,
        [Tooltip("50% -> -(0%~50%)" +
            "\n50% -> +(0%~50%)" +
            "\n@no sign change" +
            "\n@best for finetuning")]
        LightPercentage,
        [Tooltip("50% -> -(0f~.5f)" +
            "\n50% -> +(0f~.5f)")]
        LightValue,
        [Tooltip("50% -> -(0%~100%)" +
            "\n50% -> +(0%~100%)" +
            "\n@no sign change" +
            "\n@best for deep-tuning")]
        StrongPercentage,
        [Tooltip("50% -> -(0f~1f)" +
                  "\n50% -> +(0f~1f)")]
        StrongValue,
        [Tooltip("12.5% -> New value from normal distribution" +
            "\n17.5% -> Classic mutation" +
            "\n17.5% -> LightPercentage mutation" +
            "\n17.5% -> LightValue mutation" +
            "\n17.5% -> StrongPercentage mutation" +
            "\n17.5% -> StrongValue mutation")]
        Chaotic,

    }
    public enum ActivationFunctionType
    {
        //NO REAL TIME MODIFICATION
        [Tooltip("@output: 0 or 1\n" +
                 "@good for output layer - binary value")]
        BinaryStep,
        [Tooltip("@output: (0, 1)\n" +
                 "@good for output layer - good value range (positive)")]
        Sigmoid,
        [Tooltip("@output: (-1, 1)\n" +
                 "@best for output layer - good value range")]
        Tanh,
        [Tooltip("@output: [0, +inf)\n" +
                 "@good for hidden layers - low computation")]
        Relu,
        [Tooltip("@output: (-inf*,+inf)\n" +
                 "@best for hidden layers - low computation")]
        LeakyRelu,
        [Tooltip("@output: [-0.278, +inf)\n" +
                 "@smooth ReLU - higher computation")]
        Silu,
        [Tooltip("@output: [0, 1]\n" +
                 "@output activation ONLY\n" +
                 "@good for decisional output")]
        SoftMax,

    }
    public enum InitializationFunctionType
    {
        [Tooltip("@value: [-1, 1]")]
        RandomValue,
        [Tooltip("@Box-Muller method in Standard Normal Distribution")]
        StandardDistribution,
        [Tooltip("@value: average 0.673\n" +
           "@l = 0.15915f\n" +
           "@k = 1.061f\n" +
           "@z = 0.3373f")]
        Deviation1Distribution,
        [Tooltip("@value: average 0.725\n" +
            "@l = 0.15915f\n" +
            "@k = 2f\n" +
            "@z = 0.3373f")]
        Deviation2Distribution,
    }
    public enum LossFunctionType
    {
        [Tooltip("@(output - expectedOutput)^2")]
        Quadratic,
        [Tooltip("@abs(output - expectedOutput)")]
        Absolute,
        [Tooltip("@only for SoftMax outputActivation\nonly when ActionBuffer has only values of 0 with one value of 1")]
        CrossEntropy
    }
    public enum HeuristicModule
    {
        [Tooltip("@append data to the file below\n@creates a file if doesn't exist using the name from Samples Path")]
        Collect,
        [Tooltip("@use data from the file below")]
        Learn,
    }

    public enum AlgorithmType
    {
        [Tooltip("@next generation is made by selecting the fittest half and mutate it")]
        Evolutionary,
        [Tooltip("@next generation is made by combining the genes of 2 random parents\n@fittest agents are more likely to be parents")]
        Genetic,

              
    }
    public enum CrossoverType
    {
        [Tooltip("@one offspring's parameters part receive the parameters of one of parent and second part from the other")]
        OnePoint,
        [Tooltip("@each offspring's parameter receive the value of one random parent parameter")]
        Uniform,
        [Tooltip("@offspring's parameters are the average values of the parents' parameters")]
        Average,
        [Tooltip("@offspring's parameters are the weighted values of the parents' parameters\n@fitter parent has more impact on the offspring")]
        Weighted,

    }
}
