#include <iostream>
#include <string.h>
#include <cstdio>
#include <vector>
#include <iostream>
#include <fstream>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int otsu(cv::Mat &img)
{
    float histogram[256] = { 0 };
    for (int i = 0; i<img.rows; i++)
    {
        unsigned char* p = (unsigned char*)img.ptr(i);
        for (int j = 0; j<img.cols; j++)
        {
            histogram[p[j]]++;
        }
    }
    float avgValue = 0;
    int numPixel = img.cols*img.rows;
    for (int i = 0; i<256; i++)
    {
        histogram[i] = histogram[i] / numPixel;
        avgValue += i*histogram[i];
    }
    int threshold = 0;
    float gmax = 0;
    float wk = 0, uk = 0;
    for (int i = 0; i<256; i++) {
        wk += histogram[i];
        uk += i*histogram[i];
        float ut = avgValue*wk - uk;
        float g = ut*ut / (wk*(1 - wk));
        if (g > gmax)
        {
            gmax = g;
            threshold = i;
        }
    }
    return threshold;
}

cv::Mat rgb_Procession(Mat color_pic)
{
    Mat mask;
    Mat gray;
    vector<Mat> channels;
    // 把一个3通道图像转换成3个单通道图像
    split(color_pic, channels);//分离色彩通道
    gray = (2* channels.at(1) - channels.at(2) - channels.at(0));
    GaussianBlur(gray, gray, Size(3, 3), 0, 0);
    int thresh = otsu(gray);
    color_pic.copyTo(mask, gray);
    return mask;
}

cv::Mat hsv_Procession(cv::Mat color_pic)
{
    //颜色的HSV范围
    int iLowH = 10;
    int iHighH = 80;

    int iLowS = 10;
    int iHighS = 255;

    int iLowV = 10;
    int iHighV = 255;

    Mat HSVimg;
    Mat mask;
    cvtColor(color_pic, HSVimg, COLOR_BGR2HSV);//转为HSV
    Mat imgThresholded;
    inRange(HSVimg, Scalar(iLowH, iLowS, iLowV), Scalar(iHighH, iHighS, iHighV), imgThresholded);

    //开操作 (去除一些噪点)  如果二值化后图片干扰部分依然很多，增大下面的size
    Mat element = getStructuringElement(MORPH_RECT, Size(3, 3));
    morphologyEx(imgThresholded, imgThresholded, MORPH_OPEN, element);
    //闭操作 (连接一些连通域)
    morphologyEx(imgThresholded, imgThresholded, MORPH_CLOSE, element);
    color_pic.copyTo(mask, imgThresholded);
    return mask;
}

double Process(cv::Mat InputImage)
{
    cv::Mat result;
    // 提取绿色分量
    result = rgb_Procession(InputImage);
    cv::imshow("result",result);
    cv::waitKey();

    cv::Mat mask;
    cv::cvtColor(result, mask, CV_BGR2GRAY);
    // 提阈值处理
    cv::Mat mask1;
    cv::threshold(mask, mask1, 0.0, 255.0, CV_THRESH_BINARY | CV_THRESH_OTSU);
    // 查找轮廓，对应连通域
    vector<vector<cv::Point>> contours;
    cv::findContours(mask1,contours,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_NONE);
    // 寻找最大连通域
    double sumArea = 0;
    vector<cv::Point> maxContour;
    for(size_t i = 0; i < contours.size(); i++)
    {
        double area = cv::contourArea(contours[i]);
        sumArea = sumArea + area;
    }
    return sumArea;
}

int main() {
    string pattern_bmp = "../data/imgdata2/*.bmp";
    vector<cv::String> image_files;
    glob(pattern_bmp, image_files);
    if (image_files.size() == 0) {
        std::cout << "No image files[bmp]" << std::endl;
        return 0;
    }
    for (unsigned int frame = 0; frame < image_files.size(); ++frame)
    {
        Mat image = imread(image_files[frame]);
        std::cout<<Process(image)<<endl;
    }
    return 0;
}
