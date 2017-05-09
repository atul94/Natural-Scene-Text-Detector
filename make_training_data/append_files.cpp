#include <bits/stdc++.h>
#include <string>
using namespace std;
string to_string(int t)
{
	string s = "";
	while(t)
	{
		char c = ('0' + t%10);
		s = c + s;
		t /= 10;
	}
	return s;
}
int main()
{
	string s1 = "./o1/";
	ofstream myfile;
	string OUTPUT_FILE = "./new.txt";
	myfile.open(OUTPUT_FILE.c_str());
	long long l = 0;
	for(int i = 100; i <= 228; i++)
	{
		string s = s1 + to_string(i) + ".txt";
		ifstream infile(s.c_str());
		string line;
		while(getline(infile,line))
		{
			l++;
			myfile << line << "\n";
		}
		infile.close();
	}
	cout << l << "\n";
	myfile.close();
}