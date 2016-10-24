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
RNG rng(12345);
const char* INPUT_FILE;
ofstream myfile;
CvBoost adatree;
int dx[8] = {-1,0,1,1,1,0,-1,-1};
int dy[8] = {1,1,1,0,-1,-1,-1,0};
int circle_dx[12] = {-2,-2,-1,0,1,2,2,2,1,0,-1,-2};
int circle_dy[12] = {0,1,2,2,2,1,0,-1,-2,-2,-2,-1};
int outer_circle_dx[12] = {-2,-2,-1,0,1,2,2,2,1,0,-1,-2};
int outer_circle_dy[12] = {0,1,2,2,2,1,0,-1,-2,-2,-2,-1};
int inner_circle_dx[8] = {-1,-1,0,1,1,1,0,-1};
int inner_circle_dy[8] = {0,1,1,1,0,-1,-1,-1};
int mymap[12] = {0,1,1,2,3,3,4,5,5,6,7,7};

struct description
{
	int property;// SEK = 1, SBK = 2, normal = 0
	int intensity; // Dark Point  = 1, Bright Point = -1;
	int contrast; // As the name suggests
	int threshold; // For the floodfill
	int mod_same_points;
};
struct region_properties
{
	double compactness;
	Rect bounding_box;
	double convex_hull_ratio;
	double holes_area_ratio;//(convex hull)/(bounding rect)
	double CSA;
	double property; //1.0 for text -1.0 for non-text
};
int analyze_circle(const vector<int> &similar_points_index)
{
	if(similar_points_index.size() == 1)
		return 1;
	int segments = 1;
	for(int x = 1; x < similar_points_index.size(); x++)
	{
		if((similar_points_index[x] - similar_points_index[x-1]) != 1 )
			segments++;
	}
	
	if(similar_points_index[similar_points_index.size()-1] == 11 && similar_points_index[0] == 0)
		segments--;
	if(segments == 1 && similar_points_index.size() < 4)
		return 1;
	else if(segments == 2 && similar_points_index.size() < 6)
		return 2;
	else 
		return 0;
}
description check_keypoints(const Mat &bwimg, int i, int j,int edgeThreshold)
{
	vector <int> outer_circle(12);
	for(int k = 0; k < 12; k++)
		outer_circle[k] = bwimg.at<uchar>(i+outer_circle_dx[k],j+outer_circle_dy[k]);
	vector <int> inner_circle(8);
	for(int k = 0; k < 8; k++)
		inner_circle[k] = bwimg.at<uchar>(i+inner_circle_dx[k],j+inner_circle_dy[k]);
	int count_sp = 0;
	int count_dp = 0;
	int count_bp = 0;
	int dark_threshold = (int)bwimg.at<uchar>(i,j) - edgeThreshold;
	int bright_threshold = (int)bwimg.at<uchar>(i,j) + edgeThreshold;
	description flag;
	flag.property = 0; flag.intensity = 0; flag.contrast = 0; flag.threshold = 0;flag.mod_same_points = 0;
	int f = 1;
	vector <int> similar_points_index;
	vector<int> mymap(12);
	mymap[0] = 0; mymap[1] = 1; mymap[2] = 1; mymap[3] = 2; mymap[4] = 3; mymap[5] = 3; mymap[6] = 4; mymap[7] = 5; mymap[8] = 5; mymap[9] = 6; mymap[10] = 7; mymap[11] = 7; 
	for(int x = 0; x < 12; x++)
	{
		if(outer_circle[x] < bright_threshold && outer_circle[x] > dark_threshold)
		{
			if(inner_circle[mymap[x]] > dark_threshold && inner_circle[mymap[x]] < bright_threshold)
			{
				if(x < 6)
				{
					if(outer_circle[x+6] < bright_threshold && outer_circle[x+6] > dark_threshold)
					{
						f = 0;
						break;
					}
				}
				count_sp++;
				similar_points_index.push_back(x);
			}
			else
			{
				f = 0;
				break;
			}
		}
		
		else if(outer_circle[x] <= dark_threshold && count_bp == 0)
		{
			count_dp++;
			if(count_dp == 1)
				flag.threshold = outer_circle[x];
			else if(flag.threshold < outer_circle[x])
				flag.threshold = outer_circle[x];
		}
		else if(outer_circle[x] >= bright_threshold && count_dp == 0)
		{
			count_bp++;
			if (count_bp == 1)
				flag.threshold = outer_circle[x];
			else if(flag.threshold > outer_circle[x])
				flag.threshold = outer_circle[x];
		}
		else 
		{
			f = 0;
			break;
		}
	}
	if(f == 0 || count_sp > 5 || count_sp==0)
	{
		flag.threshold = 0;
		return flag;
	}
	flag.property = analyze_circle(similar_points_index);
	if(flag.property == 0)
	{
		flag.threshold = 0;
		return flag;
	}
	if(count_bp == 0)
	{
		flag.threshold++;
		flag.intensity = 1;
		flag.mod_same_points = count_sp;
		flag.contrast = (int)bwimg.at<uchar>(i,j) - *min_element(outer_circle.begin(),outer_circle.end());
		return flag;
	}
	else
	{
		flag.threshold--;
		flag.intensity = -1;
		flag.mod_same_points = count_sp;
		flag.contrast = *max_element(outer_circle.begin(),outer_circle.end()) - (int)bwimg.at<uchar>(i,j);
		return flag;
	}
}
void non_maximal_suppression(vector<vector <description> > &KPs, int x, int y)
{
	for (int i = 2; i < x - 2; i++)
	{
		for(int j = 2; j < y - 2; j++)
		{
			if(KPs[i][j].property!=0)
			{
				vector <int> max_contrast;
				for(int k = i-1; k < i + 2; k++)
				{
					for(int l = j-1; l < j + 2; l++)
					{
						
						if(KPs[k][l].property==KPs[i][j].property && KPs[k][l].intensity==KPs[i][j].intensity)
						{
							max_contrast.push_back(KPs[k][l].contrast);
						}
					}
				}
		
				if(KPs[i][j].contrast != *max_element(max_contrast.begin(),max_contrast.end()))
				{
					KPs[i][j].property = 0; KPs[i][j].intensity = 0; KPs[i][j].threshold = 0;KPs[i][j].contrast = 0;
				}
			}
		}
	}
	return;
}
vector < vector <description> > find_KeyPoints( int edgeThreshold,const Mat &bwimg)
{
	description t;
	t.property = 0;t.intensity = 0;t.contrast = 0;t.threshold = 0;
	vector < vector <description> > KeyPoints(bwimg.rows, vector <description>(bwimg.cols,t));	
	for(int i = 2; i < bwimg.rows - 2; i++)
	{
		for(int j = 2; j < bwimg.cols - 2; j++)
		{
			description flag = check_keypoints(bwimg, i, j, edgeThreshold);
			KeyPoints[i][j] = flag;
		}
	}
				
	return KeyPoints;
}
void find_position(const vector <vector <description> > &KPs, vector < Point > &position)
{
	for(int i = 2; i < KPs.size() - 2; i++)
	{
		for(int j = 2; j < KPs[0].size() - 2; j++)
			if(KPs[i][j].property!=0)
			{
				Point temp;
				temp.x = j;
				temp.y = i;
				position.push_back(temp);
			}
	}
}
void floodfill_1(vector <vector < Point > > &r,Mat &img, int x, int y, const Mat &bwimg, const vector<vector<description> > &KeyPoints, int key, int &threshold, int i, int &c)
{
	if(c > 1000)
		return;
	if(img.at<Vec3b>(x,y)[i%3]==255 && (img.at<Vec3b>(x,y)[(i+1)%3] == (23 * i)%256) && img.at<Vec3b>(x,y)[(i+2)%3]== (29 * i)%256)
		return;
	if(key == -1 && (bwimg.at<uchar>(x,y) > threshold) && c!=0)
		return;
	else if(key == 1 && (bwimg.at<uchar>(x,y) < threshold) && c!=0)
		return;
	if((key + KeyPoints[x][y].intensity) == 0)
		return;
	
	img.at<Vec3b>(x,y)[i%3] = 255;img.at<Vec3b>(x,y)[(i+1)%3] = (23 * i)%256; img.at<Vec3b>(x,y)[(i+2)%3]=(29 * i)%256;
	c++;
	Point temp;
	temp.y = x;
	temp.x = y;
	r[i].push_back(temp);
	if(x < bwimg.rows -1)
	{
		//cout << "x = " << x << " y = " << y << endl;
		floodfill_1(r,img, x+1, y, bwimg, KeyPoints, key, threshold,i,c);
	}
	if(x > 0)
	{
		//cout << "x = " << x << " y = " << y << endl;
		floodfill_1(r,img, x-1, y, bwimg, KeyPoints, key, threshold,i,c);
	}
	if(y < bwimg.cols - 1)
	{
		//cout << "x = " << x << " y = " << y << endl;
		floodfill_1(r,img, x, y+1, bwimg, KeyPoints, key, threshold,i,c);
	}
	if(y > 0)
	{
		//cout << "x = " << x << " y = " << y << endl;
		floodfill_1(r,img, x, y-1, bwimg, KeyPoints, key, threshold,i,c);
	}
	return;
}

///////////////////////////
//Euler number code
void dfs(int x,int y,int c, Mat &g, vector <vector <int> > &w)
{
    w[x][y] = c;
    for(int i=0; i<8;i++)
    {
        int nx = x+dx[i];
        int ny = y+dy[i];
        if(nx >=0 && nx < g.rows && ny >=0 && ny < g.cols)
        if(g.at<uchar>(nx,ny)==0 && !w[nx][ny]) dfs(nx,ny,c,g,w);
    }
}
double eccentricity_calculator( vector<Point> &contour )
{ 
	RotatedRect ellipse = fitEllipse(contour);
	return (double)sqrt(1 - ((ellipse.size.width / ellipse.size.height) *(ellipse.size.width / ellipse.size.height))); 
}
void euler_number(Mat &bwimg, vector <vector <Point> >  &r1, vector <vector <Point> > &r2)
{
	
	vector <Mat> matrix;
	int borderType = BORDER_CONSTANT;
	for(int i = 0; i < r1.size(); i++)
	{
		//cout << "i = " << i << "\n";
		if(r1.size() <= 10)
			continue;
		double eccentricity = 1.0;
		try
		{
			eccentricity = eccentricity_calculator(r1[i]);
			
		}
		catch(...)
		{
			continue;
		}
		if(eccentricity >= 0.995)
				continue;
		//try
		//{
		Mat matROI = Mat(r1[i]);
		Rect roi = boundingRect(r1[i]);
		Mat mask = Mat::zeros(bwimg.size(), CV_8UC1);
		drawContours(mask, r1, i, Scalar(255), CV_FILLED);
		Mat mask2 = (mask(roi));
		copyMakeBorder(mask2,mask2,1,1,1,1,borderType,Scalar(0));
		//cout << mask2.rows << " " << mask2.cols << endl;
		vector <vector <int> > w(mask2.rows, vector <int>(mask2.cols,0));
		int set = 1;
		for(int x=0; x<mask2.rows;x++)
		{
        	for(int y=0; y<mask2.cols; y++)
        	{
            	if(mask2.at<uchar>(x,y)==0 && !w[x][y])
                	dfs(x,y,set++,mask2,w);
        	}
        	if(set > 5)
        		break;
        }
        if(set > 5)
        	continue;
    	//}
    	//catch(...)
    	//{
    	//	continue;
    	//}
        //cout << "set = " << set << endl;
        //imshow("testing",mask2);
        //waitKey(0);
        r2.push_back(r1[i]);
	}
	
}
//////////////////////////////////////////////////////
///CSA
int check_circle_CSA(vector <int> &similar_points_index)
{

	int segments = 1;
	vector <int> segments_size(1,1);
	for(int i = 1; i < similar_points_index.size(); i++)
	{
		if(similar_points_index[i]-similar_points_index[i-1] == 1)
			segments_size[segments-1]++;
		else
		{
			segments++;
			segments_size.push_back(1);
		}
	}
	if(similar_points_index[similar_points_index.size()-1]==11 && similar_points_index[0]==0)
	{
		segments_size[0] = segments_size[0] + segments_size[segments-1];
		segments--;
	}
	if(segments == 2)
		return *max_element(segments_size.begin(),segments_size.begin()+segments);
	return 0;
}
void sum_CSA(int &ans,int index_x,int index_y,Mat &bwimg,Mat &mask,int p_x, int p_y)
{
	if(index_x < 0 || index_y < 0 || index_x >= bwimg.rows || index_y >= bwimg.cols)
		return;
	if(mask.at<uchar>(index_x,index_y) == 0)
		return;
	mask.at<uchar>(index_x,index_y) = 0;
	int upper_limit = bwimg.at<uchar>(index_x,index_y) + 13;
	int lower_limit = bwimg.at<uchar>(index_x,index_y) - 13;
	vector <int> similar_points_index;
	int val = 0;
	int x,y;
	for(int i = 0; i < 12; i++)
	{
		int a = index_x + circle_dx[i];
		int b = index_y + circle_dy[i];
		//cout << "a = " << a << " b = " << b << "\n";
		if(a >= 0&& b >=0 && a < bwimg.rows && b < bwimg.cols && bwimg.at<uchar>(a,b) <= upper_limit && bwimg.at<uchar>(a,b) >= lower_limit)
		{
			
			similar_points_index.push_back(i);
			if(a==p_x && b == p_y)
				continue;
			if(bwimg.at<uchar>(a,b) > val)
			{
				val = bwimg.at<uchar>(a,b);
				x = a;
				y = b;
			}
		}
	}
	if(similar_points_index.size() < 2 || similar_points_index.size() > 6)
		return;
	int f = check_circle_CSA(similar_points_index);
	if(f == 0)
		return;
	else
	{
		ans = ans + 3 * f;
		//cout << "ans = " << ans << "\n";
		sum_CSA(ans,x,y,bwimg,mask,index_x,index_y);

	}
}
double get_CSA(vector <Point> &region, vector<vector<description> > &KeyPoints, Mat &bwimg)
{
	int CSA = 0;
	int borderType = BORDER_CONSTANT;
	Mat matROI = Mat(region);
	Mat mask = Mat::zeros(bwimg.size(), CV_8UC1);
	vector <vector <Point> > rrr;rrr.push_back(region);
	drawContours(mask, rrr, 0, Scalar(255), CV_FILLED);
	for(int j = 0; j < region.size();j++)
	{
		if(KeyPoints[region[j].y][region[j].x].property == 2)
			CSA += 3 * KeyPoints[region[j].y][region[j].x].mod_same_points;
		else if(KeyPoints[region[j].y][region[j].x].property == 1)
		{
			int index_x,index_y;
			int temp_val = 0;
			int upper_limit = bwimg.at<uchar>(region[j].y,region[j].x) + 13;
			int lower_limit = bwimg.at<uchar>(region[j].y,region[j].x) - 13;
			for(int k = 0; k < 12; k++)
			{
				int a = region[j].y + circle_dx[k];
				int b = region[j].x + circle_dy[k];
				if(a >=0 && b >=0 && a < bwimg.rows && b < bwimg.cols)
				{
					int ttt = (bwimg.at<uchar>(a,b));
					if(ttt > temp_val && ttt <= upper_limit && ttt >= lower_limit)
					{
						temp_val = ttt;
						index_x = a;
						index_y = b;
					}
				}
			}
 			sum_CSA(CSA,index_x,index_y,bwimg,mask,region[j].y,region[j].x);
		}
	}
	double t  = (double)CSA * 1.0;
	return t;
}
///////////////////////////////////////////////////////
//Removing false +ve
///////////////////////////////////

double find_in_GT(Rect A,vector <Rect> &GT_bounding_box)
{
	
	for(int i = 0; i < GT_bounding_box.size(); i++)
	{
		
		Rect intersect_rect = A&GT_bounding_box[i];
		Rect total_rect = A|GT_bounding_box[i];
		double area_intersect = intersect_rect.area();
		double total_area = total_rect.area();
		if(area_intersect/total_area >= 0.5)
			return 1.0;

	}
	return -1.0;
}
double Deviation(vector<float> v, double ave)
{
    double E=0;
    double inverse = 1.0 / static_cast<double>(v.size());
    for(unsigned int i=0;i<v.size();i++)
    {
        E += pow(static_cast<double>(v[i]) - ave, 2);
    }
    return sqrt(inverse * E);
}
double stroke_width_filter(Mat &image,   vector <Point>  &R)
{
	vector <vector <Point> > r;
	r.push_back(R);
	int borderType = BORDER_CONSTANT;
	vector<Mat> regions;
    for(int i = 0; i < r.size(); i++)
    {
		Mat matROI = Mat(r[i]);
		Rect roi = boundingRect(r[i]);
        Mat mask = Mat::zeros(image.size(), CV_8UC1);
        drawContours(mask, r, i, Scalar(255), CV_FILLED);
        Mat contourRegion;
        Mat imageROI;
        image.copyTo(imageROI, mask);
        contourRegion = imageROI(roi);
		//Mat contourRegion = Mat(r[i]);
                //threshold(contourRegion,contourRegion, 127, 1, THRESH_BINARY);
        copyMakeBorder(contourRegion,contourRegion,1,1,1,1,borderType,Scalar(0));
        regions.push_back(contourRegion);
		
	}
	vector<Mat> strokeWidthFilterIdx;
	double strokeWidthThreshold = 0.65;
	for(int i = 0; i < regions.size(); i++)
	{
		Mat img = regions[i];
		//Mat negation_region_1;
		Mat distanceImage_1;
		Mat img_gray;
		Mat img_bin;
		cvtColor(img, img_gray,CV_RGB2GRAY);
		threshold (img_gray, img_bin, 127, 255, CV_THRESH_BINARY);
		cv::Mat negation_region_1 =  cv::Scalar::all(255) - img_bin;
		distanceTransform(img_bin, distanceImage_1, CV_DIST_L2, 3);
		cvtColor(img, img,CV_RGB2GRAY);
		threshold(img, img, 127, 1, THRESH_BINARY);
		Mat skel(distanceImage_1.size(), CV_8UC1, Scalar(0));
		Mat temp(distanceImage_1.size(), CV_8UC1);
		Mat element = getStructuringElement(MORPH_CROSS, cv::Size(3, 3));
		bool done;
		do
		{
		  cv::morphologyEx(img_bin, temp, cv::MORPH_OPEN, element);
          cv::bitwise_not(temp, temp);
          cv::bitwise_and(img_bin, temp, temp);
          cv::bitwise_or(skel, temp, skel);
          cv::erode(img_bin, img_bin, element);

		  double max;
		  cv::minMaxLoc(img_bin, 0, &max);
		  done = (max == 0);
		} 
		while (!done);
		vector<float> strokeWidthValues;
		Mat test;
	    vector<cv::Point> locations;
	    
	    cv::findNonZero(skel, locations);
		Mat testContour = Mat(r[i]);	
	        vector<Point> skel_regions;
		for(int row = 0; row < skel.rows; row++)
        {
            for(int col = 0; col < skel.cols; col++)
            {
                cv::Scalar intensity = skel.at<uchar>(row,col);
                    if(intensity[0])
                    {       			
                        strokeWidthValues.push_back(distanceImage_1.at<float>(row,col));
                    }
            }
        }

        if(!strokeWidthValues.size())
        	continue;
	    //imshow("skel",skel);
	    double sum = std::accumulate(strokeWidthValues.begin(), strokeWidthValues.end(), 0.0);
        double mean = sum / strokeWidthValues.size();
		//cv::bitwise_and(distanceImage_1,skel,test);
	    double stdev = Deviation(strokeWidthValues,mean);
	    //cout<<" nan ckeck Std/mean "<<stdev/mean<<" and mean "<<mean<<" std "<<stdev<<endl;
	    return stdev/mean;

	}

}

///////////////////////////////////
void calculate_properties(vector <vector < Point > > &r,vector <region_properties> &r2, vector<vector<description> > &KeyPoints, Mat &bwimg, int mul1, int mul2)
{
	
		
    for( int i = 0; i < r.size(); i++ )
    { 

    	if(r[i].size() < 5)
    		continue;
    	region_properties t;
    	double area = contourArea(r[i],false);
    	t.bounding_box =  boundingRect(r[i]);
    	double some_ratio = ((double)t.bounding_box.area())/((double)(bwimg.rows*bwimg.cols));
    	if(some_ratio < 0.0005)
    		continue;
    	vector <double> region_property(6);
    	vector <Point> contours_poly;
    	
    	//Compactness
    	Point2f center;
    	float radius;
    	vector <Point> rrr(r[i]);
    	approxPolyDP( Mat(rrr), contours_poly, 3, true);
    	minEnclosingCircle( (Mat)contours_poly, center, radius);
    	double r = (double)radius;
    	if(area == 0)
    		continue;
    	double circle_area = (22.0/7.0)*r*r;
    	t.compactness = area/(circle_area);
    	region_property[0] = t.compactness;
    	//Convex Hull Area Ratio
    	vector <Point> hull;
    	convexHull( Mat(rrr), hull, false);
    	double hull_area = contourArea(hull);
    	t.convex_hull_ratio = area/hull_area;
    	region_property[1] = t.convex_hull_ratio;
    	//Hole Area Ratio
    	t.holes_area_ratio = (1/t.convex_hull_ratio) - 1.0;
    	region_property[2] = t.holes_area_ratio;
    	//CSA
    	t.CSA = get_CSA(rrr,KeyPoints,bwimg);
    	region_property[3] = t.CSA;

    	region_property[4] = ((double)min(t.bounding_box.height,t.bounding_box.width))/((double)max(t.bounding_box.height,t.bounding_box.width));
    	region_property[5] = area/((double)(t.bounding_box.height*t.bounding_box.width));
		t.bounding_box.x = (t.bounding_box.x * mul1)/mul2;t.bounding_box.y = (t.bounding_box.y * mul1)/mul2; t.bounding_box.height = (t.bounding_box.height * mul1)/mul2; t.bounding_box.width = (t.bounding_box.width * mul1)/mul2;
        float zzz[6];
        for(int j = 0; j < 6; j++)
        	zzz[j] = region_property[j];
        Mat m(1,6,CV_32FC1,zzz);
    	float a = adatree.predict(m);
    	cout << "a = " << a << "\n";
    	if(a > 0)    
        	r2.push_back(t);
        

		
	}
    
}


///////////////////////////////////////////////
///Creating Ground Truth Bounding Boxes
int stoi(string &s)
{
	int ans = 0;
	for(int i = 0 ; i < s.size(); i++)
		ans = ans*10 + (s[i]-'0');
	return ans;
}

bool com1(pair < pair<int,int>,int> a, pair < pair<int,int>,int> b)
{
	return a.second < b.second;
}
void KeyPointsReduction(vector<vector <description> > &KeyPoints, int x, int y)
{
	//cout << "step_size = " << x/6 << " " << y/5 << "\n";
	for(int i = 0; i < x; i = i + x/6)
	{
		for(int j = 0; j < y; j = j + y/5)
		{
			vector <pair < pair<int,int>,int> > key_point_reduction;
			for(int k = i; k < min(i+x/6,x); k++)
			{
				for(int l = j; l < min(j+y/5,y); l++)
				{
					if(KeyPoints[k][l].property != 0)
						key_point_reduction.push_back(make_pair(make_pair(k,l),KeyPoints[k][l].contrast));
				}
			}
			//cout << "size = " << key_point_reduction.size() << "\n";
			if(key_point_reduction.size() > 133)
			{
				sort(key_point_reduction.begin(),key_point_reduction.end(),com1);
				for(int iii = 0; iii < key_point_reduction.size()-133; iii++)
				{
					KeyPoints[key_point_reduction[iii].first.first][key_point_reduction[iii].first.second].property = 0;
					KeyPoints[key_point_reduction[iii].first.first][key_point_reduction[iii].first.second].threshold = 0;
					KeyPoints[key_point_reduction[iii].first.first][key_point_reduction[iii].first.second].contrast = 0;
					KeyPoints[key_point_reduction[iii].first.first][key_point_reduction[iii].first.second].intensity = 0;
				}
			}
		}
	}
}
//////////////////////////////////////////////////////
int main(int argc, char const *argv[])
{
	string temp_address = "/home/atul/Desktop/textspotter/new/python/test_images/101.jpg";
	Mat im,bwim,img2;
	
	if(argc > 1){
        INPUT_FILE = argv[1];
		im = imread(argv[1], CV_LOAD_IMAGE_COLOR);
    }
	else
		im = imread(temp_address,CV_LOAD_IMAGE_COLOR);
	/*while(im.rows > 1024)
		resize(im, im, Size(), 1024.0/double(im.rows), 1024.0/double(im.rows), INTER_AREA );*/
	cvtColor(im, bwim, CV_RGB2GRAY);
	vector <Mat> images(8), bwimages(8);
	im.copyTo(images[0]);bwim.copyTo(bwimages[0]);
	//train_();
	//cout << GT_bounding_box.size() << "\n";
	for(int i = 1; i < 8; i++)
	{
		resize(images[i-1],images[i],Size(), 0.625, 0.625, INTER_AREA);
		resize(bwimages[i-1],bwimages[i],Size(), 0.625, 0.625, INTER_AREA);
	}
	vector<Rect> bounding_box_character;
	vector <region_properties> property_vector;
    string INP_STR (INPUT_FILE);
    adatree.load("./working.xml");
    int start,end;
    start = 0; end = 3;
    cout << im.rows << "\n";
    if(im.cols > 2000 || im.rows > 2000)
    {
    	start = 4;
    	end = 6;
    }
	for(int ii = start; ii < end; ii++)
	{
		Mat bwimg, img;
		bwimg = bwimages[ii]; img = images[ii];
		if(bwimg.rows < 25 || bwimg.cols < 25)
			continue;
		vector<vector<description> > KeyPoints;
		KeyPoints = find_KeyPoints(13,bwimg);
		cout << "done find_KeyPoints\n";
		non_maximal_suppression(KeyPoints,bwimg.rows,bwimg.cols);
		//cout << bwimg.rows << " " << bwimg.cols << endl;
		KeyPointsReduction(KeyPoints,bwimg.rows,bwimg.cols);
		
		
		cout << "done non_maximal_suppression\n";
	
		vector < Point > pos;
		find_position(KeyPoints,pos);
		vector <vector < Point > > regions(pos.size());
		img.copyTo(img2);
		for(int i = 0; i < pos.size(); i++)
		{
			int c = 0;
			floodfill_1(regions,img,pos[i].y,pos[i].x,bwimg,KeyPoints,KeyPoints[pos[i].y][pos[i].x].intensity,KeyPoints[pos[i].y][pos[i].x].threshold,i,c);
		}
		cout << "Done floodfill\n";
		vector <vector < Point > > regions2;
		euler_number(bwimg,regions,regions2);
		//cout << regions2.size() << "\n";
		int mul1,mul2;
		mul1 = pow(8,ii);mul2 = pow(5,ii);
		calculate_properties(regions2,property_vector, KeyPoints, bwimg,mul1,mul2);
	}
	Mat img4;
	im.copyTo(img4);
	cout << "size = " << property_vector.size() << "\n";
	cout << property_vector.size() << "\n";
	for(int i = 0; i < property_vector.size(); i++)
	{
		
		Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
		rectangle( img4, property_vector[i].bounding_box.tl(), property_vector[i].bounding_box.br(), color, 2, 8, 0 );
	}

	namedWindow("test", WINDOW_AUTOSIZE);
	imshow("test",img4);
	waitKey(0);
	//cout << OUTPUT_FILE << "\n";
	//myfile.close();
	return 0;
}
