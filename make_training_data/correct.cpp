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
	string s1 = "./o2/";
	ofstream myfile;
	string OUTPUT_FILE = "./clear_new.txt";
	myfile.open(OUTPUT_FILE.c_str());
	string s = "./new.txt";
	long long l = 0;
	ifstream infile(s.c_str());
	string line;
	while(getline(infile,line))
	{
		int c = 0;
		for(int i = 0; i < line.size(); i++)
			if(line[i] == ' ')
				c++;
		if(c == 6)
		{
			l++;
			myfile << line << "\n";
		}
	}
	infile.close();
	
	cout << l << "\n";
	myfile.close();
}