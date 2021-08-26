
//This is a Back Propagation network. It consists layers:
//100 neurons on input layer, 50 neurons on hidden layer and one
//neuron on output layer.

//This programm use function: F(NET) = sigmoid (NET), that takes values
//from 0 to +1.
//The values of the neurons in the hidden layer are continuous.
//The values of the input and output neurons are diskreet.

#include<iostream>
#include<stdlib.h>
#include<time.h>
#include<math.h>
#include<fcntl.h>
#include<sys\stat.h>
#include<io.h>

#include "C:\רשתות\be\Patterns7.dat"        //File with patterns for input end output.

#define Low           -1
#define Hi	          +1
#define zero         0

#define InputNeurons  100	  
#define HiddenNeurons 50

typedef int InArr[InputNeurons];


using namespace std;

class Data
{
private:
	int order_arr[TrainPatt];//arry with num 0-TrainPatt, each numb represent shape
	void SetUnorderedNumbers(); //Set numbers from 0 to TrainPatt randomaly in array.

public:
	InArr* Input;
	int* Output;
	int Units;     //Numbers (units) in input ( and output ) now.

	Data();
	~Data();

	//Set input and output vectors from patterns.
	bool SetInputOutput(char[][Y][X], char*, int);

	bool SetInputOutputRand(char[][Y][X], char*, int);
	//Free memory of Input and Output units
	void Reset();
};




class BackPropagationNet
{
private:
	//Input to network 
	int    InputLayer[InputNeurons];

	//Output from hidden layer -> it is input to output layer.
	float  HiddenLayer[HiddenNeurons];

	//Output of network - one neuron.
	float  OutputLayer;	                    //Takes values: -1 or +1.

	float  WeigthsOut[HiddenNeurons];
	float  WeigthsHidd[HiddenNeurons][InputNeurons];
	float  nu;                                       //Learning rate.
	float  Threshold;
	//It was error now ?. If error occured, then NetError = true, 
	//else NetError = false.
	bool  NetError;

	float RandomEqualReal(float, float);


	//Calculate output for current input 
	void CalculateOutput();

	void ItIsError(int);           //NetError = true if it was error.

	void AdjustWeigths(int);

public:
	//Initialization of weigths and variables.
	BackPropagationNet();

	//Initialize all and randomly weigths.
	void  Initialize();


	//Train network up to 90% success or up to 1000 cycles 

	bool TrainNet(Data&);

	bool TrainNetRand(Data& data_obj, char[TrainPatt][Y][X], char*, int);


	float sigmoid(float x);
	float divSigmoid(float x);
	//Testing of network . Return success percent.
	int TestNet(Data&);

	const int ReturnOutput() { return OutputLayer; };

	float LearningRate() { return nu; };
	float ThresholdValue() { return Threshold; };
};





//********************* CLASS BACKPROPAGATIONNET *******************





BackPropagationNet::BackPropagationNet()
{
	nu = 0.1f;

	srand((unsigned)time(NULL));
	Initialize();
}


//_________________________________________________________________________


void BackPropagationNet::Initialize()
{
	int i, j;

	Threshold = 1.0f;
	NetError = false;

	//Randomize weigths (initialize).
	for (i = 0; i < HiddenNeurons; i++)
		WeigthsOut[i] = RandomEqualReal(-1.0f, 1.0f);

	for (i = 0; i < HiddenNeurons; i++)
	{
		for (j = 0; j < InputNeurons; j++)
			WeigthsHidd[i][j] = RandomEqualReal(-1.0f, 1.0f);
	}
}


//_________________________________________________________________________


//Return randomaly numbers from LowN to HighN
float BackPropagationNet::RandomEqualReal(float LowN, float HighN)
{
	return ((float)rand() / RAND_MAX) * (HighN - LowN) + LowN;
}


//_________________________________________________________________________
float BackPropagationNet::sigmoid(float x) {
	return  1 / (1 + exp(-x));
}
float BackPropagationNet::divSigmoid(float x)
{
	float d = exp(-x);
	float m = ((d + 1) * (d + 1));
	return d / m;
}
void BackPropagationNet::CalculateOutput()
{
	float Sum;

	//Calculate output for hidden layer.
	for (int i = 0; i < HiddenNeurons; i++)
	{
		Sum = 0.0f;
		for (int j = 0; j < InputNeurons; j++)
		{
			Sum += WeigthsHidd[i][j] * InputLayer[j];
		}

		HiddenLayer[i] = (float)sigmoid(Sum);
	}

	//Calculate output for output layer.
	Sum = 0.0f;

	for (int n = 0; n < HiddenNeurons; n++)
		Sum += WeigthsOut[n] * HiddenLayer[n];
	float x = sigmoid(Sum);
	//Make decision about output neuron.
	if (x > 2 / 3)
		OutputLayer = 1.0f;
	else if (x < 1 / 3)
		OutputLayer = 0.5f;
	else						                     //We can  decide.
		OutputLayer = 0.0f;
}


//_________________________________________________________________________


void BackPropagationNet::ItIsError(int Target)
{
	if (((float)Target - OutputLayer))
		NetError = true;
	else
		NetError = false;
}


//__________________________________/_______________________________________

void BackPropagationNet::AdjustWeigths(int Target)
{
	int i, j;
	float hidd_deltas[HiddenNeurons], out_delta;

	//Calcilate deltas for all layers.
	out_delta = (divSigmoid(OutputLayer)) * (Target - OutputLayer);

	for (i = 0; i < HiddenNeurons; i++)
		hidd_deltas[i] = (divSigmoid(HiddenLayer[i])) * out_delta * WeigthsOut[i];

	//Change weigths.
	for (i = 0; i < HiddenNeurons; i++)
		WeigthsOut[i] = WeigthsOut[i] + (nu * out_delta * HiddenLayer[i]);

	for (i = 0; i < HiddenNeurons; i++)
	{
		for (j = 0; j < InputNeurons; j++)
			WeigthsHidd[i][j] = WeigthsHidd[i][j] +
			(nu * hidd_deltas[i] * InputLayer[j]);
	}
}


//_________________________________________________________________________
bool BackPropagationNet::TrainNetRand(Data& data_obj, char[TrainPatt][Y][X], char*, int)
{
	int Error, j, loop = 0, Success;

	cout << endl;
	cout << "   --------------------------------------------------------";
	cout << endl << endl;
	cout << "                       TRAINING NETWORK" << endl << endl;
	cout << "   --------------------------------------------------------";
	cout << endl << endl;

	do
	{
		Error = 0;
		loop++;

		cout << "Threshold =    " << Threshold << endl;

		data_obj.SetInputOutputRand(TrainingInput, TrainingOutput, TrainPatt);

		//Printing the number of loop.
		if (loop < 10)
			cout << "Training loop:  " << loop << "       ...   ";
		if (loop >= 10 && loop < 100)
			cout << "Training loop:  " << loop << "      ...   ";
		else if (loop >= 100 && loop < 1000)
			cout << "Training loop:  " << loop << "     ...   ";


		//Train network (do one cycle).
		for (int i = 0; i < data_obj.Units; i++)
		{
			//Set current input.
			for (j = 0; j < InputNeurons; j++)
				InputLayer[j] = data_obj.Input[i][j];

			CalculateOutput();
			ItIsError(data_obj.Output[i]);

			//If it was error, change weigths (Error = sum of errors in
			//one cycle of train).
			if (NetError)
			{
				Error++;
				AdjustWeigths(data_obj.Output[i]);
			}

		}

		Success = ((data_obj.Units - Error) * 100) / data_obj.Units;
		cout << Success << " %   success" << endl << endl;



	} while (Success < 90 && loop <= 1000);

	if (loop > 1000)
	{
		cout << "Training of network failure !" << endl;
		return false;
	}
	else
		return true;


}

bool BackPropagationNet::TrainNet(Data& data_obj)
{
	int Error, j, loop = 0, Success;

	cout << endl;
	cout << "   --------------------------------------------------------";
	cout << endl << endl;
	cout << "                       TRAINING NETWORK" << endl << endl;
	cout << "   --------------------------------------------------------";
	cout << endl << endl;

	do
	{
		Error = 0;
		loop++;

		cout << "Threshold =    " << Threshold << endl;

		//Printing the number of loop.
		if (loop < 10)
			cout << "Training loop:  " << loop << "       ...   ";
		if (loop >= 10 && loop < 100)
			cout << "Training loop:  " << loop << "      ...   ";
		else if (loop >= 100 && loop < 1000)
			cout << "Training loop:  " << loop << "     ...   ";


		//Train network (do one cycle).
		for (int i = 0; i < data_obj.Units; i++)
		{
			//Set current input.
			for (j = 0; j < InputNeurons; j++)
				InputLayer[j] = data_obj.Input[i][j];

			CalculateOutput();
			ItIsError(data_obj.Output[i]);

			//If it was error, change weigths (Error = sum of errors in
			//one cycle of train).
			if (NetError)
			{
				Error++;
				AdjustWeigths(data_obj.Output[i]);
			}

		}

		Success = ((data_obj.Units - Error) * 100) / data_obj.Units;
		cout << Success << " %   success" << endl << endl;



	} while (Success < 90 && loop <= 1000);

	if (loop > 1000)
	{
		cout << "Training of network failure !" << endl;
		return false;
	}
	else
		return true;

}


//_________________________________________________________________________

int BackPropagationNet::TestNet(Data& data_obj)
{
	int Error = 0, j, Success;

	cout << endl << endl << endl;
	cout << "---------------------------------------------------------------------";
	cout << endl << endl;
	cout << "                    TEST NETWORK" << endl << endl;
	cout << "---------------------------------------------------------------------";
	cout << endl << endl;

	cout << "Test network    ...  ";

	//Train network (do one cycle).
	for (int i = 0; i < data_obj.Units; i++)
	{
		//Set current input.
		for (j = 0; j < InputNeurons; j++)
			InputLayer[j] = data_obj.Input[i][j];

		CalculateOutput();
		ItIsError(data_obj.Output[i]);

		//Error = sum of errors in this one cycle of test.
		if (NetError)
			Error++;
	}

	Success = ((data_obj.Units - Error) * 100) / data_obj.Units;
	cout << Success << "%   success" << endl;

	return Success;
}





//************************ CLASS DATA *************************************





Data::Data()
{
	Units = 0;
}


//_________________________________________________________________________


Data::~Data()
{
	Reset();
}


//_________________________________________________________________________


void Data::Reset()
{
	Units = 0;
	delete[] Input;
	delete[] Output;
}


//_________________________________________________________________________


bool Data::SetInputOutput(char In[][Y][X], char* Out, int num_patterns)
{
	int n, i, j;

	if (Units != num_patterns)
	{
		if (Units)
			Reset();

		if (!(Input = new InArr[num_patterns]))
		{
			cout << "Insufficient memory for Input" << endl;
			return false;
		}

		if (!(Output = new int[num_patterns]))
		{
			cout << "Insufficient memory for Output" << endl;
			delete[] Input;
			return false;
		}

		Units = num_patterns;
	}

	for (n = 0; n < Units; n++)                         //Set input vectors.
	{
		for (i = 0; i < Y; i++)
		{
			for (j = 0; j < (X - 1); j++)
				Input[n][i * (X - 1) + j] = (In[n][i][j] == '*') ? Hi : Low;
		}
	}

	//Set corresponding to input expected output.
	for (i = 0; i < Units; i++)
	{
		if (Out[i] == '-')
		{
			Output[i] = Hi;
		}
		else if (Out[i] == '+')
		{
			Output[i] = zero;
		}
		else Output[i] = 0.5f;
	}

	return true;
}

//_______________________________________________________
bool Data::SetInputOutputRand(char In[][Y][X], char* Out, int num_patterns)
{
	int n, i, j;

	if (Units != num_patterns)
	{
		if (Units)
			Reset();

		if (!(Input = new InArr[num_patterns]))
		{
			cout << "Insufficient memory for Input" << endl;
			return false;
		}

		if (!(Output = new int[num_patterns]))
		{
			cout << "Insufficient memory for Output" << endl;
			delete[] Input;
			return false;
		}

		Units = num_patterns;
	}

	SetUnorderedNumbers();

	cout << endl << "The order of the train:   ";
	for (i = 0; i < TrainPatt; i++)
		cout << order_arr[i] << ", ";
	cout << endl << endl;

	for (n = 0; n < Units; n++)                         //Set input vectors.
	{
		for (i = 0; i < Y; i++)
		{
			for (j = 0; j < (X - 1); j++)
				Input[n][i * (X - 1) + j] = (In[order_arr[n]][i][j] == '*') ? Hi : Low;
		}
	}

	//Set corresponding to input expected output.
	for (i = 0; i < Units; i++)
	{
		if (Out[order_arr[i]] == '-')
		{
			Output[i] = Hi;
		}
		else if (Out[order_arr[i]] == '+')
		{
			Output[i] = zero;
		}
		else Output[i] = 0.5f;
	}

	return true;
}

//____________________________________
void Data::SetUnorderedNumbers()
{
	int number, index;

	for (int i = 0; i < TrainPatt; i++)                      //Initialize array.
		order_arr[i] = -1;

	for (number = 0; number < TrainPatt; number++)
	{
		index = rand() % TrainPatt;
		if (order_arr[index] == -1)         //If the place is empty.
		{
			order_arr[index] = number;
		}

		else      //If place arr[index] is not empty, then find next
		{		    //empty place.
			while (order_arr[index] != -1)
			{
				index++;
				index = index % TrainPatt;
			}

			order_arr[index] = number;       //We finded empty place.
		}
	}
}



//***************************** MAIN **************************************


void main()
{
	Data data_obj;
	BackPropagationNet back_prop_obj;
	bool flag;

	cout << "Back Propagation Network" << endl << endl;


	//TRAINING NETWORK .

	back_prop_obj.Initialize();


	//if (!data_obj.SetInputOutput(TrainingInput, TrainingOutput, TrainPatt))
	//	return;

	//while (!(flag = back_prop_obj.TrainNet(data_obj)))
	//{
	//	back_prop_obj.Initialize();
	//	return;

	//}

	////TEST NETWORK.

	//if (!data_obj.SetInputOutput(TestInput, TestOutput, TestPatt))
	//	return;

	//back_prop_obj.TestNet(data_obj);


//________________________________________________/
	//TRAIN NETRAND

	if (!data_obj.SetInputOutputRand(TrainingInput, TrainingOutput, TrainPatt))
		return;

	while (!(flag = back_prop_obj.TrainNetRand(data_obj, TestInput, TestOutput, TestPatt)))
	{
		back_prop_obj.Initialize();
		return;

	}

	//TEST NETWORK.

	if (!data_obj.SetInputOutput(TestInput, TestOutput, TestPatt))
		return;

	back_prop_obj.TestNet(data_obj);
}



