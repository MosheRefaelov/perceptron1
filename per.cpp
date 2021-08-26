#include<iostream>
#include<stdlib.h>
#include<time.h>
#include<string.h>
#include "../../../../../../רשתות/data1.dat"

using namespace std;
typedef int InArr10[DataInputs];
typedef int  OutArr4[DataOutputs];

class Data
{
private:
	int order_arr[3];//arry with num 0-2, each numb represent shape
	void SetUnorderedNumbers(); //Set numbers from 0 to 2 randomaly in array.

public:
	//In binary  Hi = 1 and Low = 0.
	  //In bipolar Hi = +1 and Low = -1.
	int Hi;
	int Low;

	InArr10* Input;

	OutArr4* Output;
	int Units;
	Data();
	~Data();
	void ChangeSystem(const char*);//Change System to bipolar or to binary.

	//Set 3 inputs for network and set 3 outputs expected.
	bool  SetInputOutputSeq(char[Numbers][Y][X], char[Numbers][DataOutputs + 1]);

	//set randomly 3 inputs for network and set 3 outputs expected to this inputs.
	bool SetInputOutputRand(char[Numbers][Y][X], char[Numbers][DataOutputs + 1]);

	//Set 9 inputs: 3 shapes 100% correct and 2 sets ( 2 x 3 )
	//shapes with 15% error every shape ( from 100 units ).

	//Set 9 inputs: 3 sets ( 3 x 3 ) shapes with 15% error every
	//shapes.
	bool SetInputOutputError(char[Numbers][Y][X], char[Numbers][Y][X],
		char[Numbers][Y][X], char[Numbers][DataOutputs + 1]);

	//Free memory of Input and Output units
	void Reset();
};

class PerceptronNet
{
private:
	int    InputLayer[DataInputs];          //Input to network.
	int    OutputLayer[DataOutputs];        //Output of network.
	float  Weigths[DataOutputs][DataInputs]; //synaptic weights  
	int Error[DataOutputs];   //
	float eta;                                //Learning rate.
	int Threshold;                           //Threshold value.
	//It was error now ?. If error occured even though one output
		//neuron, then NetError = true, else NetError = false.
	bool NetError;

public:

	//Initialization of weigths and variables.
	PerceptronNet();

	//Initialize arrays by 0 and set randomly weigths.
	void Initialize();
	float RandomEqualReal(float, float);

	//Calculate output for current input ( bipolar or binary ).
	void CalculateOutput(int, int);

	void ComputeOutputError(int[DataOutputs+1]);

	//To correct weigths use the formula:
	//Weigths[j][i]=Weigths[j][i]+eta*(DesireOut[j]-Output[j])*Input[i].
	void AdjustWeigths();

	//Training of network up to 100% success or 40 cycles sequentialy.
		//Return true if we get 100% success in training, or false else.
	bool TrainNetSeq(Data&);

	//Training network up to 100% success or 40 cycles randomaly.
	//Return true if we gwt 100% success in training, or false else.
	bool TrainNetRand(Data& data_obj, char[Numbers][Y][X],
		char[Numbers][DataOutputs+1]);

	//Testing of network. Return success percent.
	int TestNet(Data&);

	void ChangeParametrs(float, int);  //Change value of parametrs.

	const int* ReturnOutput() { return OutputLayer; };
	float LearningRate() { return eta; };
	int ThresholdValue() { return Threshold; };

};


/////////////////
PerceptronNet::PerceptronNet()
{
	eta = 0.1f;
	Threshold = 0;

	Initialize();
	srand((unsigned)time(NULL));//time(0)
}


// Initialize the net
void PerceptronNet::Initialize()
{
	int i, j, n;

	NetError = false;

	//Randomize weigths (initialize).
	for (i = 0; i < DataOutputs; i++)
	{
		for (j = 0; j < DataInputs; j++)
			Weigths[i][j] = RandomEqualReal(-0.5f, 0.5f);
	}


	for (i = 0; i < DataInputs; i++)
		InputLayer[i] = 0;

	for (j = 0; j < DataOutputs; j++)
		OutputLayer[j] = 0;

	for (n = 0; n < DataOutputs; n++)
		Error[n] = 0;
}

///
float PerceptronNet::RandomEqualReal(float Low, float High)
{
	return ((float)rand() / RAND_MAX) * (High - Low) + Low;
}

void PerceptronNet::CalculateOutput(int Hi, int Low)
{
	float Sum;

	for (int i = 0; i < DataOutputs; i++)
	{
		Sum = 0.0f;
		for (int j = 0; j < DataInputs; j++)
		{
			Sum += Weigths[i][j] * InputLayer[j];
		}

		if (Sum >= Threshold)
			OutputLayer[i] = Hi;

		else                     //Sum < Threshold.
			OutputLayer[i] = Low;
	}
}
void PerceptronNet::ComputeOutputError(int CorrectOutput[DataOutputs])
{
	NetError = false;

	for (int i = 0; i < DataOutputs; i++)
	{
		Error[i] = CorrectOutput[i] - OutputLayer[i];

		if (Error[i] != 0 && NetError == false)
			NetError = true;
	}
}

void PerceptronNet::AdjustWeigths()
{
	for (int i = 0; i < DataOutputs; i++)
	{
		for (int j = 0; j < DataInputs; j++)
			Weigths[i][j] = Weigths[i][j] + eta * Error[i] * InputLayer[j];
	}
}

////////////////////////////////
bool PerceptronNet::TrainNetSeq(Data& data_obj)
{
	int Error, j, loop = 0, Success;

	do
	{
		Error = 0;
		loop++;

		if (loop < 10)
			cout << "Training loop:  " << loop << "    ...   ";
		else
			cout << "Training loop:  " << loop << "   ...   ";

		//Train network (do one cycle).
		for (int i = 0; i < data_obj.Units; i++)
		{
			//Set current input.
			for (j = 0; j < DataInputs; j++)
				InputLayer[j] = data_obj.Input[i][j];

			CalculateOutput(data_obj.Hi, data_obj.Low);
			ComputeOutputError(data_obj.Output[i]);

			//If it was error, change weigths (Error = sum of errors in one
				//cycle of train).
			if (NetError)
			{
				Error++;
				AdjustWeigths();
			}
		}

		Success = ((data_obj.Units - Error) * 100) / data_obj.Units;
		cout << Success << " %   success" << endl << endl;

	} while (Success < 100 && loop <= 40);

	if (loop > 40)
	{
		cout << "Training of network failure !" << endl;
		return false;
	}
	else
		return true;

}


//_________________________________________________________________________


bool PerceptronNet::TrainNetRand(Data& data_obj,
	char InputData[Numbers][Y][X],
	char OutputData[Numbers][DataOutputs+1 ])
{
	int Error, j, loop = 0, Success;

	do
	{
		Error = 0;
		loop++;

		data_obj.SetInputOutputRand(InputData, OutputData);

		if (loop < 10)
			cout << "Training loop:  " << loop << "    ...   ";
		else
			cout << "Training loop:  " << loop << "   ...   ";

		//Train network (do one cycle).
		for (int i = 0; i < data_obj.Units; i++)
		{
			//Set current input.
			for (j = 0; j < DataInputs; j++)
				InputLayer[j] = data_obj.Input[i][j];

			CalculateOutput(data_obj.Hi, data_obj.Low);
			ComputeOutputError(data_obj.Output[i]);

			//If it was error, change weigths (Error = sum of errors in one
			//cycle of train).
			if (NetError)
			{
				Error++;
				AdjustWeigths();
			}
		}

		Success = ((data_obj.Units - Error) * 100) / data_obj.Units;
		cout << Success << " %   success" << endl << endl;

	} while (Success < 100 && loop <= 40);

	if (loop > 40)
	{
		cout << "Training of network failure !" << endl;
		return false;
	}
	else
		return true;
}





//_________________________________________________________________________


int PerceptronNet::TestNet(Data& data_obj)
{
	int Error = 0, j, Success;

	cout << "Test network    ...  ";

	//Train network (do one cycle).
	for (int i = 0; i < data_obj.Units; i++)
	{
		//Set current input.
		for (j = 0; j < DataInputs; j++)
			InputLayer[j] = data_obj.Input[i][j];

		CalculateOutput(data_obj.Hi, data_obj.Low);
		ComputeOutputError(data_obj.Output[i]);

		//If it was error, change weigths (Error = sum of errors in one
			//cycle of train).
		if (NetError)
			Error++;
	}

	Success = ((data_obj.Units - Error) * 100) / data_obj.Units;
	cout << Success << "%   success" << endl;

	return Success;
}

void PerceptronNet::ChangeParametrs(float nuu, int Th)
{
	eta = nuu;
	Threshold = Th;
}




//************************ CLASS DATA *************************************

Data::Data()
{
	Units = 0;
	ChangeSystem("binary");

	for (int i = 0; i < 3; i++)
		order_arr[i] = -1;
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


void Data::SetUnorderedNumbers()
{
	int number, index;

	for (int i = 0; i < 3; i++)                      //Initialize array.
		order_arr[i] = -1;

	for (number = 0; number < 3; number++)
	{
		index = rand() % 3;
		if (order_arr[index] == -1)         //If the place is empty.
		{
			order_arr[index] = number;
		}

		else      //If place arr[index] is not empty, then find next
		{		    //empty place.
			while (order_arr[index] != -1)
			{
				index++;
				index = index % 3;
			}

			order_arr[index] = number;       //We finded empty place.
		}
	}
}

//_________________________________________________________________________


bool Data::SetInputOutputSeq(char In[Numbers][Y][X],
	char Out[Numbers][DataOutputs + 1])
{
	int n, i, j;
	//In binary : Hi = 1,  Low = 0.
	//In bipolar: Hi = +1, Low = -1.
	if (Units != 3)
	{
		if (Units)
			Reset();

		if (!(Input = new InArr10[3]))
		{
			cout << "Insufficient memory for Input" << endl;
			return false;
		}

		if (!(Output = new OutArr4[3]))
		{
			cout << "Insufficient memory for Output" << endl;
			delete[] Input;
			return false;
		}

		Units = 3;
	}

	//Set sequently input (bipolar or binary)
	for (n = 0; n < Units; n++)
	{
		for (i = 0; i < Y; i++)
		{
			for (j = 0; j < (X - 1); j++)
				Input[n][i * (X - 1) + j] = (In[n][i][j] == '*') ? Hi : Low;
		}
	}

	//Set corresponding to input expected output (bipolar or binary)
	for (i = 0; i < Units; i++)
	{
		for (j = 0; j < DataOutputs; j++)
			Output[i][j] = (Out[i][j] == '*') ? Hi : Low;
	}

	return true;
}



bool Data::SetInputOutputRand(char In[Numbers][Y][X],
	char Out[Numbers][DataOutputs + 1])
{
	int n, i, j;

	if (Units != 3)
	{
		if (Units)
			Reset();

		if (!(Input = new InArr10[3]))
		{
			cout << "Insufficient memory for Input" << endl;
			return false;
		}

		if (!(Output = new OutArr4[3]))
		{
			cout << "Insufficient memory for Output" << endl;
			delete[] Input;
			return false;
		}

		Units = 3;
	}


	SetUnorderedNumbers();

	cout << endl << "The order of the train:   ";
	for (i = 0; i < 3; i++)
		cout << order_arr[i] << ", ";
	cout << endl << endl;

	//Set randomly given input (bipolar or binary)
	for (n = 0; n < Units; n++)
	{
		for (i = 0; i < Y; i++)
		{
			for (j = 0; j < (X - 1); j++)
				Input[n][i * (X - 1) + j] = (In[order_arr[n]][i][j] == '*') ? Hi : Low;
		}
	}

	//Set corresponding to input expected output (bipolar or binary)
	for (i = 0; i < Units; i++)
	{
		for (j = 0; j < DataOutputs; j++)
			Output[i][j] = (Out[order_arr[i]][j] == '*') ? Hi : Low;
	}

	return true;
}


//_________________________________________________________________________


bool Data::SetInputOutputError(char In1[Numbers][Y][X],
	char In2[Numbers][Y][X], char In3[Numbers][Y][X],
	char Out[Numbers][DataOutputs + 1])
{
	int n, i, j;;

	if (Units != 9)
	{
		if (Units)
			Reset();

		if (!(Input = new InArr10[9]))
		{
			cout << "Insufficient memory for Input" << endl;
			return false;
		}

		if (!(Output = new OutArr4[9]))
		{
			cout << "Insufficient memory for Output" << endl;
			delete[] Input;
			return false;
		}

		Units = 9;
	}


	//Set input ( bipolar or binary ).
	for (n = 0; n < Numbers; n++)
	{
		for (i = 0; i < Y; i++)
		{
			for (j = 0; j < (X - 1); j++)
				Input[n][i * (X - 1) + j] = (In1[n][i][j] == '*') ? Hi : Low;
		}
	}

	for (n = 0; n < Numbers; n++)
	{
		for (i = 0; i < Y; i++)
		{
			for (j = 0; j < (X - 1); j++)
				Input[n + 3][i * (X - 1) + j] = (In2[n][i][j] == '*') ? Hi : Low;
		}
	}


	for (n = 0; n < Numbers; n++)
	{
		for (i = 0; i < Y; i++)
		{
			for (j = 0; j < (X - 1); j++)
				Input[n + 6][i * (X - 1) + j] = (In3[n][i][j] == '*') ? Hi : Low;
		}
	}

	//Set output ( bipolar or binary ).
	for (i = 0; i < Numbers; i++)
	{
		for (j = 0; j < DataOutputs; j++)
			Output[i][j] = (Out[i][j] == '*') ? Hi : Low;
	}


	for (i = 0; i < Numbers; i++)
	{
		for (j = 0; j < DataOutputs; j++)
			Output[i + 3][j] = (Out[i][j] == '*') ? Hi : Low;
	}


	for (i = 0; i < Numbers; i++)
	{
		for (j = 0; j < DataOutputs; j++)
			Output[i + 6][j] = (Out[i][j] == '*') ? Hi : Low;
	}

	return true;
}


//__________________________________________________________________________


void Data::ChangeSystem(const char* system)
{
	if (!strcmp(system, "binary"))   //If binary system.
	{
		Hi = 1;
		Low = 0;
	}

	else if (!strcmp(system, "bipolar"))  //If bipolar system.
	{
		Hi = +1;
		Low = -1;
	}

	else
	{
		cout << "Not correct system type !" << endl;
		cout << "System type have to be binary or bipolar." << endl;
	}
}






//***************************** MAIN **************************************


int main()
{
	//******************* PART 1 of project *******************************


	Data data_obj;
	PerceptronNet percep_obj;

	char quit;
	float learning_rate;//eta
	int threshold_value;

	cout << endl;
	cout << "   --------------------------------------------------------" << endl << endl;
	cout << "            TRAINING NETWORK: RANDOMALY BINARY" << endl << endl;
	cout << "   --------------------------------------------------------" << endl << endl;
	cout << "Learning rate   = " << percep_obj.LearningRate() << endl;
	cout << "Threshold value = " << percep_obj.ThresholdValue() << endl << endl;

	data_obj.ChangeSystem("binary");

	while (!percep_obj.TrainNetRand(data_obj, InputPattern, OutputPattern)) //While not 100% sucess in
	{										   //training network, train
		percep_obj.Initialize();			   //network newly.
		percep_obj.ChangeParametrs(percep_obj.RandomEqualReal(0.0001f, 0.999f),
			percep_obj.ThresholdValue());

		cout << "----------------------------------------------------" << endl;
		cout << "Training newly of network" << endl;
		cout << "Learning rate   = " << percep_obj.LearningRate() << endl;
		cout << "Threshold value = " << percep_obj.ThresholdValue() << endl;

	}


	cout << endl << endl << endl;
	cout << "   --------------------------------------------------------" << endl << endl;
	cout << "            TRAINING NETWORK: SEQUENTIALY BINARY" << endl << endl;
	cout << "   --------------------------------------------------------" << endl << endl;
	percep_obj.ChangeParametrs(0.1f, 0);
	percep_obj.Initialize();	      //Randomize weigths.
	cout << "Learning rate   = " << percep_obj.LearningRate() << endl;
	cout << "Threshold value = " << percep_obj.ThresholdValue() << endl << endl;


	if (!data_obj.SetInputOutputSeq(InputPattern, OutputPattern))
		return 0;

	while (!percep_obj.TrainNetSeq(data_obj)) //While not 100% sucess in
	{										   //training network, train
		percep_obj.Initialize();			   //network newly.
		percep_obj.ChangeParametrs(percep_obj.RandomEqualReal(0.0001f, 0.999f),
			percep_obj.ThresholdValue());

		cout << "----------------------------------------------------" << endl;
		cout << "Training newly of network" << endl;
		cout << "Learning rate   = " << percep_obj.LearningRate() << endl;
		cout << "Threshold value = " << percep_obj.ThresholdValue() << endl;

	}

	//******************* PART 2 of project *******************************


	cout << endl << endl << endl;
	cout << "---------------------------------------------------------------------" << endl << endl;
	cout << "   TEST NETWORK ON SET OF 3 CORUPTED PATTERNS (SEQUENTIALY BINARY)" << endl << endl;
	cout << "---------------------------------------------------------------------" << endl << endl;
	if (!data_obj.SetInputOutputSeq(InputErrPattern1, OutputPattern))
		return 0;

	percep_obj.TestNet(data_obj);


	//******************* PART 3 of project *******************************


	cout << endl << endl << endl;
	cout << "   ------------------------------------------------------------------" << endl << endl;
	cout << "       TRAINING NETWORK: SEQUENTIALY BINARY ON SET OF 9 PATTERNS" << endl << endl;
	cout << "   ------------------------------------------------------------------" << endl << endl;

	percep_obj.ChangeParametrs(0.1f, 0);
	percep_obj.Initialize();
	cout << "Learning rate   = " << percep_obj.LearningRate() << endl;
	cout << "Threshold value = " << percep_obj.ThresholdValue() << endl << endl;

	if (!data_obj.SetInputOutputError(InputPattern, InputErrPattern2,
		InputErrPattern3, OutputPattern))
		return 0;

	while (!percep_obj.TrainNetSeq(data_obj)) //While not 100% sucess in
	{										   //training network, train
		percep_obj.Initialize();			   //network newly.
		percep_obj.ChangeParametrs(percep_obj.RandomEqualReal(0.0001f, 0.999f),
			percep_obj.ThresholdValue());
		cout << "----------------------------------------------------" << endl;
		cout << "Training newly of network" << endl;
		cout << "Learning rate   = " << percep_obj.LearningRate() << endl;
		cout << "Threshold value = " << percep_obj.ThresholdValue() << endl;
	}


	//******************* PART 4 of project *******************************


	cout << endl << endl << endl;
	cout << "   ------------------------------------------------------------------" << endl << endl;
	cout << "    TEST NETWORK ON SET OF 10 CORUPTED NUMBERS (SEQUENTIALY BINARY)" << endl << endl;
	cout << "   ------------------------------------------------------------------" << endl << endl;

	if (!data_obj.SetInputOutputSeq(InputErrPattern1, OutputPattern))
		return 0;

	percep_obj.TestNet(data_obj);


	//******************* PART 4 of project *******************************

	cout << endl << endl << endl;
	cout << "   ------------------------------------------------------------------" << endl << endl;
	cout << "               TESTING WITH CHANGING PARAMETRS (IN TRAINING)" << endl << endl;
	cout << "   ------------------------------------------------------------------" << endl << endl;
	do
	{
		cout << "Please enter new learning rate (from 0 to 1)" << endl;
		cin >> learning_rate;
		cout << "Please enter new threshold value" << endl;
		cin >> threshold_value;
		cout << endl;

		percep_obj.ChangeParametrs(learning_rate, threshold_value);
		percep_obj.Initialize();

		cout << "Learning rate   = " << percep_obj.LearningRate() << endl;
		cout << "Threshold value = " << percep_obj.ThresholdValue() << endl << endl;

		if (!data_obj.SetInputOutputSeq(InputPattern, OutputPattern))
			return 0;

		while (!percep_obj.TrainNetSeq(data_obj)) //While not 100% sucess in
		{										   //training network, train
			percep_obj.Initialize();			   //network newly.
			percep_obj.ChangeParametrs(percep_obj.RandomEqualReal(0.0001f, 0.999f),
				percep_obj.ThresholdValue());
			cout << "----------------------------------------------------" << endl;
			cout << "Training newly of network" << endl;
			cout << "Learning rate   = " << percep_obj.LearningRate() << endl;
			cout << "Threshold value = " << percep_obj.ThresholdValue() << endl;
		}

		cout << "Do You want to quit ?" << endl;
		cout << "Please enter: Y to quit or N to continue" << endl;
		cin >> quit;
		cout << endl;
		cout << "------------------------------------------------------------------" << endl << endl;
	} while (quit != 'Y');
}


