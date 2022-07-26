using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using System.IO;
using System.Text;
using System.Linq;

public class Agent : MonoBehaviour
{
    /// <summary>
    ///  base.Update() must be placed in override
    ///  base.Awake() must be placed in override
    /// </summary>
    [Header("Agent Properties-------------------------------")]
    public BehaviourType behaviour = BehaviourType.Default;
    [SerializeField, Range(1,15), Tooltip("The number of Inputs that the Agent will receive (-1,1)")] private int spaceSize = 1;
    [SerializeField, Range(1, 15), Tooltip("The number of Outputs that the Agent will return (-1,1)")] private int actionSize = 1;
    [SerializeField] protected NeuralNetwork network = null;
    [SerializeField] bool createOtherNeuralNetwork = false;
    [SerializeField] bool clearAndMakeANewNeuralNetwork = false;
    [Space, Header("Network Properties-----------------------------")]
    [SerializeField, Tooltip("If path!=null, it means it has a model assigned")] string path = null;
    public int[] layersFormat = null;


    float[] inputs = null;
    protected virtual void Awake()
    {
        if(path!=null && path != "")
        {
            SetNetworkFromFile(path, ref this.network);
        }
    }
    //---------------------BASE FUNCTIONS------------------//
    protected virtual void Update()
    {
        if(createOtherNeuralNetwork == true)
        {
            CreateNeuralNetwork(layersFormat,true);
            createOtherNeuralNetwork = false;
        }
        if(clearAndMakeANewNeuralNetwork == true)
        {
            CreateNeuralNetwork(layersFormat, false);
            clearAndMakeANewNeuralNetwork = false;
        }
        if (behaviour == BehaviourType.Inference)
            Inference(true);
        else if (behaviour == BehaviourType.Heuristic)
            Heuristic();
        else 
            Default();




    }


    //-----------------------FUNCTIONS-------------------//
    private void CreateNeuralNetwork(int[] layers, bool needTxtFile)
    {
        if(needTxtFile)
        CreateTextFileForNN();
        
        layers[0] = spaceSize;
        layers[layers.Length - 1] = actionSize;
        network = new NeuralNetwork(layers);
        inputs = new float[spaceSize];
        for (int i = 0; i < inputs.Length; i++)
        {
            inputs[i] = 0;
        }
        UpdateTextFile();

    }
    private void CreateTextFileForNN()
    {
        StringBuilder SBtxtNN = new StringBuilder();
        SBtxtNN.Append(Application.streamingAssetsPath);
        SBtxtNN.Append("/Neural_Networks/");
        SBtxtNN.Append("NeuralNetworkID");
        SBtxtNN.Append((this.gameObject.GetInstanceID() * -1).ToString());
        SBtxtNN.Append(".txt");
        path = SBtxtNN.ToString();

        if (!File.Exists(path))
            File.Create(path);
        else
        {

            Debug.Log("There is already a copy of this neural network!");

        }
    }
    private void AssignNeuralNetworkFrom(string path)
    {
        //This Function Assigns a TXT file to this Agent, he can modify it
        this.path = path;
        this.network = null;
        SetNetworkFromFile(path, ref this.network);
    }
    private void AssignCopyOfNeuralNetworkFrom(string path)
    {
        string newPath = path.Substring(0, path.Length - 4) + "Copy.txt";
        if(File.Exists(newPath))
        {
            File.Create(newPath);
        }
        string contents = File.ReadAllText(path);
        File.WriteAllText(newPath, contents);
        this.path = newPath;


        //This Function Make a copy of the TXT file and assigns it to this Agent, he can modify the copy not the original

    }
    private void UpdateTextFile()
    {
        ///summary
        /// []layers
        /// [][][]weights
        File.WriteAllText(path, string.Empty);
        File.AppendAllText(path, string.Join(",", network.GetLayers()));
        File.AppendAllText(path, "\n");
        float[][][] weights = network.GetWeights();
        foreach (float[][] layWeights in weights)
        {
            foreach (float[] neurWeights in layWeights)
            {
                File.AppendAllText(path, string.Join(",", neurWeights) + ",");
                
            }
            File.AppendAllText(path, "\b\n");
        }
    }
    private void SetNetworkFromFile(string path, ref NeuralNetwork network)
    {
        if(new FileInfo(path).Length == 0)
        {
            Debug.Log("The Neural Network at the path " + path + " was empty, but we initialized a new Neural Network in it!");
            CreateNeuralNetwork(layersFormat, false);
            return;
        }
        List<string> fileLines = File.ReadAllLines(path).ToList();
        ///
        ///   The strings read from the txt may not be separed corectly. For Example in case of floats, the Line string[] has +2 more elements,
        ///   so in this case 
        ///

        
             //Instatiate Neural Network
            string[] line1 = fileLines[0].Split(',');
            int[] line1_32 = new int[line1.Length];//THIS LINE REPREZENTS THE LAYERS []
            ConvertStrArrToIntArr(line1, ref line1_32);
            network = new NeuralNetwork(line1_32);

            //Update Property in Inspector
             spaceSize = line1_32[0];
             actionSize = line1_32[line1_32.Length-1];
            layersFormat = new int[line1_32.Length];
            for (int i = 0; i < line1_32.Length; i++)
            {
            layersFormat[i] = line1_32[i];
            }


            //Copy weights data
            List<float[][]> weightsList = new List<float[][]>();
            for (int i = 1; i < fileLines.Count; i++)
            {
                //One line here are the weights on a single layer
                List<float[]> weightsOnLayer = new List<float[]>();

                string[] line = fileLines[i].Split(',');
                float[] line_32 = new float[line.Length];
                ConvertStrArrToFloatArr(line,ref line_32);
                //Pana aici sunt bine

                //This array must be devided depeding on the previous layer number of neurons
                int numNeurOnPrevLayer = line1_32[i - 1];
                float[] weightsOnNeuron = new float[numNeurOnPrevLayer]; 
                int count = 0;

                
                for (int j = 0; j < line_32.Length; j++)
                {

                       if (count < numNeurOnPrevLayer)
                       {   weightsOnNeuron[count] = line_32[j];
                           count += 1;
                       }
                       else
                       {
                           weightsOnLayer.Add(weightsOnNeuron);
                           weightsOnNeuron = null;
                           weightsOnNeuron = new float[numNeurOnPrevLayer];
                           count = 0;
                           weightsOnNeuron[count] = line_32[j];
                           count++;

                       }
                   
               
                }
                
                weightsList.Add(weightsOnLayer.ToArray());
            }
            network.SetWeights(weightsList.ToArray());//Final set

        
    }
   

    private void Default()
    {
        if (network == null)
            Heuristic();
        else
            Inference(false);
    }
    protected virtual void Heuristic()
    {

    }
    private void Inference(bool canTrain)
    {
        if(network == null)
        {
            Debug.Log("Agent " + this.name + " cannot inference because it doesn't have a NeuralNetwork assigned! The Behaviour Mode was changed to Default");
            behaviour = BehaviourType.Default;
                return;
        }
        UpdateInputs();
        if(inputs != null)
        OnOutputsReceived(PullOutputsFrom(inputs));

        if(canTrain)
             network.MutateWeights();
    }

    protected virtual void UpdateInputs()
    {
        inputs = new float[network.GetLayers().Length]; //Initialize with 0
        //THEN this function somehow must be overrided in player controller




    }
    private float[] PullOutputsFrom(float[] inputs)
    {
        if (network == null)
            return null;
        else
            return network.FeedForward(inputs);
    }
    protected virtual void OnOutputsReceived(float[] outputs) // Inference ONLY
    {

    }
    protected void AddReward(float reward)
    {
        if(network == null)
        {
            Debug.LogError("Cannot modify reward because neural network is null");
            return;
        }
        else
            network.AddFitness(reward);

    }
    protected void SetReward(float reward)
    {
        if (network == null)
        {
            Debug.LogError("Cannot modify reward because neural network is null");
            return;
        }
        else
            network.SetFitness(reward);
    }




    //-----------------------Optional Methods-------------------//
    private void ConvertStrArrToIntArr(string[] str, ref int[] arr)
    {
        for (int i = 0; i < arr.Length; i++)
        {
            arr[i] = int.Parse(str[i]);

        }
    }
    private void ConvertStrArrToFloatArr(string[] str, ref float[] arr)
    {
        for (int i = 0; i < arr.Length; i++)
        {
            try
            {
                arr[i] = float.Parse(str[i]);
            }
            catch(System.Exception e)
            {
                Debug.LogError(str[i] + " : " + e);
            }

        }
    }
}




public enum BehaviourType
{
    [Tooltip("if nn -> nn control without training\n" +
             "else  -> heuristic control")]
    Default,
    [Tooltip("Keyboard Control from Player")]
    Heuristic,
    [Tooltip("Self Control using the Network & Training")]
    Inference
}