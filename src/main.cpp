/*
 * @Author: your name
 * @Date: 2022-01-10 23:53:55
 * @LastEditTime: 2022-01-12 22:35:28
 * @LastEditors: Please set LastEditors
 * @Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 * @FilePath: /yolov5-deepsort-tensorrt/src/main.cpp
 */
#include<iostream>
#include "manager.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include <chrono>
#include <map>
#include <cmath>
#include <time.h>
using namespace cv;




int main(){
	// calculate every person's (id,(up_num,down_num,average_x,average_y))
	map<int,vector<int>> personstate;
	map<int,int> classidmap;
	bool is_first = true;
	char* yolo_engine = "/work/tools/yolov5-deepsort-tensorrt/models/yolov5/yolov5s.engine";
	char* sort_engine = "/work/tools/yolov5-deepsort-tensorrt/models/deepsort/yolosort/deepsort.engine";
	float conf_thre = 0.4;
	Trtyolosort yosort(yolo_engine,sort_engine);
	VideoCapture capture;
	cv::Mat frame;
	// frame = capture.open("/work/tools/yolov5-deepsort-tensorrt/videos/demo2.avi");
	frame = capture.open("/work/tools/FairMOT/videos/MOT16-03.mp4");
	// frame = capture.open(0);
	if (!capture.isOpened()){
		std::cout<<"can not open"<<std::endl;
		return -1 ;
	}
	capture.read(frame);
	std::vector<DetectBox> det;
	auto start_draw_time = std::chrono::system_clock::now();
	
	clock_t start_draw,end_draw;
	start_draw = clock();
	int i = 0;
	while(capture.read(frame)){
		if (i%3==0){
		//std::cout<<"origin img size:"<<frame.cols<<" "<<frame.rows<<std::endl;
		auto start = std::chrono::system_clock::now();
		yosort.TrtDetect(frame,conf_thre,det);
		auto end = std::chrono::system_clock::now();
		int delay_infer = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		std::cout  << "delay_infer:" << delay_infer << "ms" << std::endl;
		// yosort.showDetection(frame,det);
		}
		i++;
	}
	capture.release();
	return 0;
	
}
