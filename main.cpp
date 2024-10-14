#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>

int main() {

    std::string prototxt = "models/deploy.prototxt";
    std::string model = "models/mobilenet_iter_73000.caffemodel";
    cv::dnn::Net net = cv::dnn::readNetFromCaffe(prototxt, model);


    std::string videoPath = "resources/video.mp4";
    cv::VideoCapture cap(videoPath);
    if (!cap.isOpened()) {
        std::cerr << "Can't open video file." << std::endl;
        return -1;
    }

    cv::Mat frame, resizedFrame;
    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        cv::Mat blob = cv::dnn::blobFromImage(frame, 0.007843, cv::Size(300, 300), 127.5, false, false);
        net.setInput(blob);

        cv::Mat output = net.forward();

        cv::Mat detectionMat(output.size[2], output.size[3], CV_32F, output.ptr<float>());
        for (int i = 0; i < detectionMat.rows; i++) {
            float confidence = detectionMat.at<float>(i, 2);
            int objectClass = static_cast<int>(detectionMat.at<float>(i, 1)); // Класс объекта

            if (confidence > 0.2 && objectClass == 15) { // фильтр только для людей (class 15)
                int xLeftBottom = static_cast<int>(detectionMat.at<float>(i, 3) * frame.cols);
                int yLeftBottom = static_cast<int>(detectionMat.at<float>(i, 4) * frame.rows);
                int xRightTop = static_cast<int>(detectionMat.at<float>(i, 5) * frame.cols);
                int yRightTop = static_cast<int>(detectionMat.at<float>(i, 6) * frame.rows);

                cv::rectangle(frame, cv::Point(xLeftBottom, yLeftBottom), cv::Point(xRightTop, yRightTop), cv::Scalar(0, 255, 0), 2);
            }
        }

        cv::resize(frame, resizedFrame, cv::Size(), 0.2, 0.2);

        cv::imshow("Pedestrian Detection", resizedFrame);

        if (cv::waitKey(1) == 'q') break;
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
