#include <iostream>
#include <opencv2/opencv.hpp>

#include <omp.h>

using namespace cv;

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
    Mat output = Mat::zeros(input.size(), CV_8UC3); 

    Mat kernel2;
    rotate(kernel[0], kernel2, ROTATE_90_COUNTERCLOCKWISE);

    for(int y = 1; y < input.cols - 1; y++) {
        for(int x = 1; x < input.rows - 1; x++) {
            float f_x = filterPixel(input, x, y, kernel2, factor);
            float f_y = filterPixel(input, x, y, kernel[0], factor);
            float magnitude = sqrt(f_x * f_x + f_y * f_y);

            if (magnitude > threshold) {
                float slope = atan2(f_y, f_x);

                float normalized_slope = (slope + M_PI) / (2 * M_PI);

                Vec3b color;
                if (normalized_slope <= 0.25)
                    color = Vec3b(255, 0, 0);
                else if (normalized_slope <= 0.5)
                    color = Vec3b(0, 255, 0);
                else if (normalized_slope <= 0.75)
                    color = Vec3b(0, 0, 255); 
                else
                    color = Vec3b(255, 255, 0); 

                output.at<Vec3b>(x, y) = color;
            }
        }
    }
    return output;
}


Mat convFilterMulti(const Mat &input, std::vector<Mat> kernel, const int &factor, const float &threshold) {
    Mat output = Mat::zeros(input.size(), CV_8UC3);

    Mat kernel3, kernel4;
    rotate(kernel[0], kernel3, ROTATE_90_CLOCKWISE);
    rotate(kernel[1], kernel4, ROTATE_90_CLOCKWISE);

    for(int y = 1; y < input.cols - 1; y++) {
        for(int x = 1; x < input.rows - 1; x++) {
            float g1 = filterPixel(input, x, y, kernel[0], factor);
            float g2 = filterPixel(input, x, y, kernel[1], factor);
            float g3 = filterPixel(input, x, y, kernel3, factor);
            float g4 = filterPixel(input, x, y, kernel4, factor);
            
            float maxGradient = std::max(std::max(std::abs(g1), std::abs(g2)), std::max(std::abs(g3), std::abs(g4)));
            
            if (maxGradient > threshold) {
                float direction = 0.0;
                if (maxGradient == std::abs(g1))
                    direction = 0.0;
                else if (maxGradient == std::abs(g2))
                    direction = M_PI / 4.0;
                else if (maxGradient == std::abs(g3))
                    direction = M_PI / 2.0;
                else if (maxGradient == std::abs(g4))
                    direction = 3 * M_PI / 4.0;

                float degrees = direction * 180.0 / M_PI;
                
                Vec3b color;
                if (degrees == 0.0)
                    color = Vec3b(0, 0, 255); 
                else if (degrees == 45.0)
                    color = Vec3b(0, 255, 0); 
                else if (degrees == 90.0)
                    color = Vec3b(255, 0, 0); 
                else if (degrees == 135.0)
                    color = Vec3b(255, 255, 0);
                
                output.at<Vec3b>(x, y) = color;
            }       
        }
    }
    return output;
}





float norm(float n) {
    return sqrt(n*n);
}


float calculerModule(float gradient_x, float gradient_y) {
    return sqrt(gradient_x * gradient_x + gradient_y * gradient_y);
}

float calculerPente(float gradient_x, float gradient_y) {
    return atan2(gradient_y, gradient_x) * 180 / CV_PI;
}

void initAndShow(const std::string &link, bool multi) {
    Mat img = imread(link, IMREAD_GRAYSCALE);
    Mat (*convolution)(const Mat&, std::vector<Mat>, const int&, const float&) = multi == false ? &convFilter : &convFilterMulti;
    
    std::vector<std::vector<Mat>> kernels = {
        {PREWITT, PREWITT2},
        {SOBEL, SOBEL2},
        {KIRSCH, KIRSCH2}
    };

    std::cout << "Image utilisée : " << link;
    if (multi == false) {
        std::cout << " avec filtrage bi-directionnel" << std::endl;
    } else {
        std::cout << " avec filtrage multi-directionnel" << std::endl;
    }

    int normFilterPR = normalizeFilter(PREWITT);
    Mat output_prewitt = convolution(img, kernels[0], normFilterPR, 40.0);

    int normFilterSO = normalizeFilter(SOBEL);
    Mat output_sobel = convolution(img, kernels[1], normFilterSO, 40.0);

    int normFilterKI = normalizeFilter(KIRSCH);
    Mat output_kirsch = convolution(img, kernels[2], normFilterKI, 40.0);

    imshow("image initiale", img);
    imshow("image filtrée avec Prewitt", output_prewitt);
    imshow("image filtrée avec Sobel", output_sobel);
    imshow("image filtrée avec Kirsch", output_kirsch);
}


int main(int argc, char ** argv) {
    #pragma omp parallel

    std::map<std::string, std::string> links {
        {"s" , "ExempleSimple.jpg"},
        {"c" , "Cathedrale-Lyon.jpg"},
        {"l" , "lena.png"},
    };


    std::cout << "Lancez ce programme avec comme argument '-m' pour le filtrage multi-directionnel" << std::endl;
    std::cout << "Vous pouvez également mettre {s|c|l} comme second pour afficher respectivement 'ExempleSimple', 'Cathedrale-Lyon', ou Lena (par défaut : lena)" << std::endl;
    std::cout << "Exemple : ./TP1 -m s pour afficher l'ExempleSimple en multi-directionnel" << std::endl;

    if(argc == 1) {
        initAndShow("lena.png", false);
    } else if (argc == 2 && strcmp(argv[1], "-m") == 0) {
        initAndShow("lena.png", true);
    } else if (argc == 2 && strcmp(argv[1], "s") == 0 || strcmp(argv[1], "c") == 0 || strcmp(argv[1], "l") == 0) {
        initAndShow(links[argv[1]], false);
    } else if (argc == 3 && strcmp(argv[1], "-m") == 0) {
        initAndShow(links[argv[2]], true);
    } else if (argc == 3 && strcmp(argv[2], "-m") == 0) {
        initAndShow(links[argv[1]], true);
    } else std::cerr << "ERREUR : trop d'arguments" << std::endl;

    waitKey();

    return 0;
}