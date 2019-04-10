#include "opencv2\opencv.hpp"
#include <iostream>
using namespace cv;


void setUnknown(Mat& inputOutput, const Mat& unknown)
{
	for (int i = 0; i < inputOutput.rows; i++)
	{
		for (int j = 0; j < inputOutput.cols; j++)
		{
			if (unknown.at<uint8_t>(i, j) == 255)
			{
				inputOutput.at<int>(i, j) = 0;
			}
		}
	}
}

void drawRedoutline(Mat& inputOutput, const Mat& markers) 
{
	const int channels = inputOutput.channels();
	switch (channels)
	{
		case 1: // Daca imaginea este grayscale o convertim la BGR pentru a desena
		{
			Mat temp(inputOutput.size(), CV_8UC3);
			cvtColor(inputOutput, temp, COLOR_GRAY2BGR);
			for (int i = 0; i < inputOutput.rows; i++)
			{
				for (int j = 0; j < inputOutput.cols; j++)
				{
					if (markers.at<int>(i, j) == -1)
					{
						temp.at<Vec3b>(i, j)[0] = 0;
						temp.at<Vec3b>(i, j)[1] = 0;
						temp.at<Vec3b>(i, j)[2] = 255;
					}
				}
			}
			inputOutput = temp;
		}
		case 3:
		{
			for (int i = 0; i < inputOutput.rows; i++)
			{
				for (int j = 0; j < inputOutput.cols; j++)
				{
					if (markers.at<int>(i, j) == -1)
					{
						inputOutput.at<Vec3b>(i, j)[0] = 0;
						inputOutput.at<Vec3b>(i, j)[1] = 0;
						inputOutput.at<Vec3b>(i, j)[2] = 255;
					}
				}
			}
		}
	}
}

void roi(Mat& source)
{
	Mat image = source;

	Mat mask = Mat::zeros(image.size(), CV_8UC1);
	Point circleCenter(mask.cols / 2, mask.rows / 3);
	int radius = mask.cols / 2;
	circle(mask, circleCenter, radius, CV_RGB(255, 255, 255), FILLED);
	
	/*Mat imagePart = Mat::zeros(image.size(), image.type());
	image.copyTo(imagePart, mask);*/

	int cut = image.rows - radius / 2;
	int latime = image.cols;

	// Cut si latime determina height-ul respectiv width-ul pentru ROI
	Mat roi;
	image(Rect(Point(0, cut), Point(latime, 0))).copyTo(roi);
	imshow("Original", image);
	imshow("Region of Interest", roi);
	source = roi;
	
	waitKey();
}

void main()
{

	double t = (double)getTickCount();
	
	// input
	Mat original = imread("brain.png");

	// Functia o sa taie din height-ul imagini
	//roi(original);
	
	Mat originalCopy, gray, grayCopy;
	Mat thresh;
	original.copyTo(originalCopy);
	cvtColor(original, gray, COLOR_BGR2GRAY);
	gray.copyTo(grayCopy);

	// Binarizare otsu
	threshold(gray, thresh, 0, 255, THRESH_BINARY | THRESH_OTSU);
	
	// Noise removal
	Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
	Mat opening;
	morphologyEx(thresh, opening, MORPH_OPEN, kernel, Point(-1, -1), 2);

	// Backround
	Mat background;
	dilate(opening, background, kernel, Point(-1, -1), 3);

	// Foreground
	Mat dist_transform;
	Mat foreground;
	double min, max;
	distanceTransform(opening, dist_transform, DIST_L2, 3); // euclidiana
	minMaxLoc(dist_transform, &min, &max);
	// Parametrul 3 determina valoarea limita, el trebuie schimbat pentru diferite poze
	threshold(dist_transform, foreground, 0.1*max, 255, 0); 
	// Normalizam pentru vizualizare
	normalize(dist_transform, dist_transform, 0, 1.0, NORM_MINMAX);


	// Zona Unknown
	foreground.convertTo(foreground, CV_8UC1);
	Mat unknown;
	subtract(background, foreground, unknown);

	// Markers
	// Zonele pe care le cunoastem cu siguranta (foreground si backround) sunt etichetate cu intregi diferiti
	// Iar zona pe care nu o cunoastem o etichetam cu 0
	Mat markers;
	Mat visualMakers;
	Mat colorMap;
	// Eticheteaza backround-ul cu 0 si restul obiectelor cu intregi incepand de la 1
	connectedComponents(foreground, markers);
	// Insa functia watershed considera 0 zona necunoscuta
	// Adaugam 1 la markers pentru ca backround-ul sa fie etichetat cu 1
	// Iar apoi setam zonele necunoscute cu 0
	markers = markers + 1;
	setUnknown(markers, unknown);
	// Vizualizare markere
	markers.convertTo(visualMakers, CV_8UC1);
	normalize(visualMakers, colorMap, 0, 255, NORM_MINMAX, CV_8U);
	normalize(visualMakers, visualMakers, 0, 255, NORM_MINMAX, CV_8U);
	applyColorMap(colorMap, colorMap, COLORMAP_JET);
	
	// Watershed
	watershed(original, markers);
	drawRedoutline(originalCopy, markers);
	drawRedoutline(grayCopy, markers);

	t = ((double)getTickCount() - t) / getTickFrequency();
	std::cout << "Timp trecut in secunde: " << t << std::endl;

	/*imshow("Coins", original);
	waitKey();
	imshow("Gray", gray);
	waitKey();*/
	imshow("Threshold", thresh);
	waitKey();
	/*imshow("Opening", opening);
	waitKey();
	imshow("Background", background);
	waitKey();
	imshow("Foreground",foreground);
	waitKey();
	imshow("Unknown", unknown);
	waitKey();*/
	imshow("Distance Transform", dist_transform);
	waitKey();
	//imshow("visualMakers", visualMakers);
	imshow("colorMap", colorMap);
	waitKey();

	/*//Re-vizualizare markere
	markers.convertTo(visualMakers, CV_8UC1);
	normalize(visualMakers, colorMap, 0, 255, NORM_MINMAX, CV_8U);
	normalize(visualMakers, visualMakers, 0, 255, NORM_MINMAX, CV_8U);
	applyColorMap(colorMap, colorMap, COLORMAP_JET);*/

	/*namedWindow("watershedColormap", WINDOW_FREERATIO);
	namedWindow("Watershed", WINDOW_FREERATIO);*/
	namedWindow("WatershedGray", WINDOW_FREERATIO);
	/*imshow("watershedColormap", colorMap);
	waitKey();
	imshow("Watershed", originalCopy);
	waitKey();*/
	imshow("WatershedGray", grayCopy);
	waitKey();
}