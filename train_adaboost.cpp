#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>
#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>
#include <string>
#include <vector>
#include <algorithm>
#include <numeric>
#include <math.h>
#include <fstream>
#include <iomanip>
using namespace std;
using namespace cv;

int main()
{
	string s = "./t.txt";
	ifstream infile(s.c_str());
	vector <vector <float> > training_data;
	vector <float> label;
	while(!infile.eof())
	{
		vector <float> t(6);
		float x;
		infile >> t[0] >> t[1] >> t[2] >> t[3] >> t[4] >> t[5] >> x;
		if(x != 1.0 && x != -1.0)
			continue;
		training_data.push_back(t);
		label.push_back(x);
	}
	float t[training_data.size()][6];
	float l[label.size()];
	for(int i = 0; i < training_data.size(); i++)
	{
		for(int j = 0; j < 6; j++)
			t[i][j] = training_data[i][j];
		l[i] = label[i];
	}
	cout << "done\n";
	CvBoost adatree;
	CvBoostParams boostingParameters;

	boostingParameters.boost_type       = CvBoost::GENTLE;
	boostingParameters.weak_count       = 100;
	boostingParameters.weight_trim_rate = 0.95;
	boostingParameters.max_depth        = 25;
	boostingParameters.use_surrogates   = false;
	boostingParameters.max_categories   = 2;
	boostingParameters.min_sample_count = 100;
	string filename = "./trained_model2.xml";
	int ss = training_data.size();
	Mat labelsMat(ss,1,CV_32FC1,l);
	//cout << t[175704][3] << "\n";
	Mat trainingDataMat(ss,6,CV_32FC1,t);
	adatree.train(trainingDataMat,CV_ROW_SAMPLE,labelsMat,Mat(),Mat(),Mat(),Mat(),boostingParameters,false);
	long long correct = 0; long long wrong = 0;
	adatree.save("./working.xml");
	long long fp,tp,fn,tn;
	fp=tp=fn=tn=0;
	for(int i = 0 ; i < ss; i++)
	{
		float tt[6];
		for(int j = 0; j < 6; j++)
			tt[j] = training_data[i][j];
		Mat check(1,6,CV_32FC1,tt);
		float x = adatree.predict(check);
		if(x == label[i])
		{
			if(x == 1.0)
				tp++;
			else
				tn++;
			correct++;
		}
		else
		{
			if(x == 1.0)
				fp++;
			else
				fn++;
			wrong++;
		}
	}
	cout << "correct = " << correct << " wrong = " << wrong << "\n";
	cout << "tp = " << tp << " tn = " << tn << " fp = " << fp << " fn = " << fn << "\n"; 
 	double accuracy = (double)correct/ss;
	cout << "accuracy = " << accuracy << "\n";
	cout << "done\n";
}