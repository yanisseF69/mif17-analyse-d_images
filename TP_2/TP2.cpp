#include <iostream>
#include <opencv2/opencv.hpp>
#include <omp.h>

using namespace cv;
using namespace std;

// const string INPUT_IMAGE_PATH = "Droites_simples.png";
const string INPUT_IMAGE_PATH = "../TP_1/ExempleSimple.jpg";
const int CANNY_THRESHOLD1 = 50;
const int CANNY_THRESHOLD2 = 150;
const double HOUGH_RHO = 1;
const double HOUGH_THETA = CV_PI / 180;
const int HOUGH_THRESHOLD = 80; // j'adapte cette variable pour changer la sensibilité de la detetction des lignes
const Scalar LINE_COLOR = Scalar(0, 0, 255);
const int LINE_THICKNESS = 2;

// Le filtrage Prewitt 
std::vector<float> PR_DATA = { 
    -1, 0, 1,
    -1, 0, 1,
    -1, 0, 1
};

std::vector<float> PR_DATA2 = { 
     0, 1, 1,
    -1, 0, 1,
    -1, -1, 0
};

const Mat PREWITT(3, 3, CV_32F, PR_DATA.data());
const Mat PREWITT2(3, 3, CV_32F, PR_DATA2.data());

// Le filtrage Sobel 
std::vector<float> SO_DATA = {
    -1, 0, 1,
    -2, 0, 2,
    -1, 0, 1
};

std::vector<float> SO_DATA2 = {
     0, 1, 2,
    -1, 0, 1,
    -2, -1, 0
};
const Mat SOBEL(3, 3, CV_32F, SO_DATA.data());
const Mat SOBEL2(3, 3, CV_32F, SO_DATA2.data());


// Le filtrage Kirsch
std::vector<float> KI_DATA = {
    -3, -3, 5,
    -3, 0, 5,
    -3, -3, 5
};

std::vector<float> KI_DATA2 = {
    -3, 5, 5,
    -3, 0, 5,
    -3,-3,-3
};

const Mat KIRSCH(3, 3, CV_32F, KI_DATA.data());
const Mat KIRSCH2(3, 3, CV_32F, KI_DATA2.data());

int normalizeFilter(Mat_<int> kernel) {
    int factor = 0;
    for(int y = 0; y < kernel.cols; y++) {
        for(int x = 0; x < kernel.rows; x++) {
            if(kernel.at<int>(x, y) > 0) factor += kernel.at<int>(x, y);
        }
    }

    return factor;

}

float filterPixel(const Mat & input, int x, int y, const Mat& kernel, const int & factor) {
    double res = 0;

    for (int j = y - 1; j <= y + 1; j++) {
        for (int i = x - 1; i <= x + 1; i++) {
            if (i >= 0 && i < input.rows && j >= 0 && j < input.cols) {
                res += kernel.at<_Float32>(i - (x - 1), j - (y - 1)) * static_cast<int>(input.at<uchar>(i, j));
            }
        }
    }

    res /= factor;

    return static_cast<float>(res); 
}

Mat convFilter(const Mat &input, std::vector<Mat> kernel, const int &factor, const float &threshold) {
    Mat output = Mat::zeros(input.size(), input.type()); 

    Mat kernel2;
    rotate(kernel[0], kernel2, ROTATE_90_COUNTERCLOCKWISE);

    for(int y = 1; y < input.cols - 1; y++) {
        for(int x = 1; x < input.rows - 1; x++) {
            float f_x = filterPixel(input, x, y, kernel2, factor);
            float f_y = filterPixel(input, x, y, kernel[0], factor);
            float magnitude = sqrt(f_x * f_x + f_y * f_y);
            output.at<uchar>(x, y) = magnitude;
        }
    }
    return output;
}



void HoughLinesP(cv::Mat image, std::vector<cv::Vec4i>& lines, double rho, double theta, int threshold, double minLineLength, double maxLineGap) {
    // Dimensions de l'image
    int width = image.cols;
    int height = image.rows;

    // Calcul de la longueur maximale de la ligne (la diagonale de l'image)
    double max_rho = sqrt(width * width + height * height);
    int rhoBins = cvRound(max_rho / rho) * 2;  // + et - valeurs de rho
    int thetaBins = cvRound(CV_PI / theta);

    // Accumulateur pour la transformée de Hough
    std::vector<std::vector<int>> accumulator(rhoBins, std::vector<int>(thetaBins, 0));

    // Remplissage de l'accumulateur
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            if (image.at<uchar>(y, x) > 0) {  // Assurez-vous que l'image est en niveau de gris et binarisée
                for (int t = 0; t < thetaBins; ++t) {
                    double currentTheta = t * theta;
                    double rhoValue = x * cos(currentTheta) + y * sin(currentTheta);
                    int rhoIndex = cvRound(rhoValue / rho + (rhoBins / 2));
                    accumulator[rhoIndex][t]++;
                }
            }
        }
    }

    std::vector<std::pair<int, int>> lineCandidates;
    for (int r = 0; r < rhoBins; ++r) {
        for (int t = 0; t < thetaBins; ++t) {
            if (accumulator[r][t] >= threshold) {
                lineCandidates.push_back(std::make_pair(r, t));
            }
        }
    }

    // Convertir les candidats en segments de ligne
    for (const auto& candidate : lineCandidates) {
        int r = candidate.first;
        int t = candidate.second;
        double currentTheta = t * theta;
        double rhoValue = (r - (rhoBins / 2)) * rho;

        // Trouver les extrémités du segment de ligne en parcourant tous les pixels
        cv::Point pt1, pt2;
        double a = cos(currentTheta), b = sin(currentTheta);
        double x0 = a * rhoValue, y0 = b * rhoValue;

        pt1.x = cvRound(x0 + 500 *(-b));
        pt1.y = cvRound(y0 + 500 *(a));
        pt2.x = cvRound(x0 - 500 *(-b));
        pt2.y = cvRound(y0 - 500 *(a));

        // Vérifier si les segments de ligne correspondent à la longueur minimale et à l'écart maximal spécifié
        double lineLength = sqrt(pow(pt2.x - pt1.x, 2) + pow(pt2.y - pt1.y, 2));
        if (lineLength >= minLineLength) {
            lines.push_back(cv::Vec4i(pt1.x, pt1.y, pt2.x, pt2.y));
        }
    }
}



void convertPolarToCartesian(const double rho, const double theta, cv::Point& pt1, cv::Point& pt2, int width, int height) {
    double a = cos(theta), b = sin(theta);
    double x0 = a * rho, y0 = b * rho;
    pt1.x = cvRound(x0 + 1000*(-b));
    pt1.y = cvRound(y0 + 1000*(a));
    pt2.x = cvRound(x0 - 1000*(-b));
    pt2.y = cvRound(y0 - 1000*(a));
}

void drawLines(std::vector<cv::Vec4i>& lines, cv::Mat& image) {
    int r, g, b;
    for (size_t i = 0; i < lines.size(); i++) {
        cv::Point pt1(lines[i][0], lines[i][1]), pt2(lines[i][2], lines[i][3]);
        r = rand() % 256;
        g = rand() % 256;
        b = rand() % 256;
        cv::line(image, pt1, pt2, Scalar(r, g, b), 1, LINE_AA);
    }
}

void houghCircles(const Mat& edges, vector<Vec3i> &circles, int minRadius, int maxRadius, int threshold) {

    int rows = edges.rows;
    int cols = edges.cols;
    vector<vector<vector<int>>> accumulator(maxRadius - minRadius, vector<vector<int>>(rows, vector<int>(cols)));

    for (int y = 0; y < edges.rows; y++) {
        for (int x = 0; x < edges.cols; x++) {
            if (edges.at<uchar>(y, x) > 20) {
                for (int r = minRadius; r < maxRadius; r++){
                    for (int b = y - r - 5; b < y + r + 5; b++) {
                        for (int a = x - r - 5; a < x + r + 5; a++) {
                            if (a >= 0 && b >= 0 && a < cols && b < rows) {
                                int distanceAuCentre = (x - a) * (x - a) + (y - b) * (y - b);
                                if (distanceAuCentre == r * r) {
                                    accumulator[r-minRadius][b][a]++;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    for (int r = 0; r < maxRadius - minRadius; r++){
        for (int b = 0; b < rows; b++) {
            for (int a = 0; a < cols; a++) {
                if(accumulator[r][b][a] > threshold) {
                    circles.push_back(Vec3i(a, b, r + minRadius));
                }
            }
        }
    }
}


void drawCircles(Mat& image, const vector<Vec3i>& circles) {
    for (size_t i = 0; i < circles.size(); i++) {
        Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
        int radius = cvRound(circles[i][2]);
        circle(image, center, radius, Scalar(0, 0, 255), 3, LINE_AA);
    }
}

int main(int argc, char ** argv) {

    Mat image = imread(INPUT_IMAGE_PATH, IMREAD_GRAYSCALE);

    std::vector<std::vector<Mat>> kernels = {
        {PREWITT, PREWITT2},
        {SOBEL, SOBEL2},
        {KIRSCH, KIRSCH2}
    };

    imshow("Image de base", image);


    int normFilter = normalizeFilter(SOBEL);
    Mat edges = convFilter(image, kernels[1], normFilter, 40.0);
    imshow("contours", edges);


    std::vector<Vec4i> lines;
    HoughLinesP(edges, lines, 1, CV_PI/180, 135, 0, 2000);

    std::vector<Vec3i> circles;
    #pragma omp parallel shared(circles)
    {
        houghCircles(edges, circles, 70, 110, image.rows / 20);
    }
    

    cvtColor(image, image, COLOR_GRAY2RGB);

    drawLines(lines, image);
    drawCircles(image, circles);

    cv::imshow("Lignes et cercles détectés", image);
    cv::waitKey(0);
    return 0;

}