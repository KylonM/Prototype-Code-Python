#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <thread>
#include <mutex>
#include <atomic>
#include <vector>
#include <iostream>
#include <string>


// continuously read camera frames in a background thread
class CameraStream {
private:
    cv::VideoCapture cap;
    cv::Mat current_frame;
    std::thread worker_thread;
    std::mutex mtx;
    std::atomic<bool> stopped;
    bool has_frame;

    void update() {
        while (!stopped) {
            cv::Mat temp_frame;
            bool ret = cap.read(temp_frame);
            if (!ret || !cap.isOpened()) {
                stopped = true;
                break;
            }
            {
                std::lock_guard<std::mutex> lock(mtx);
                current_frame = temp_frame.clone();
                has_frame = true;
            }
        }
    }

public:
    CameraStream(int camera_index) : stopped(false), has_frame(false) {
        cap.open(camera_index);
        cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
        
        // Grab the initial frame
        if (cap.isOpened()) {
            cv::Mat temp;
            if (cap.read(temp)) {
                std::lock_guard<std::mutex> lock(mtx);
                current_frame = temp.clone();
                has_frame = true;
            }
        }
    }

    CameraStream& start() {
        worker_thread = std::thread(&CameraStream::update, this);
        return *this;
    }

    bool read(cv::Mat& out_frame) {
        std::lock_guard<std::mutex> lock(mtx);
        if (!has_frame || current_frame.empty()) return false;
        current_frame.copyTo(out_frame);
        return true;
    }

    void release() {
        stopped = true;
        if (worker_thread.joinable()) {
            worker_thread.join();
        }
        if (cap.isOpened()) {
            cap.release();
        }
    }

    ~CameraStream() {
        release();
    }
};

int main() {
    //get input for number of cameras
    int num_cameras = 0;
    while (true) {
        std::cout << "Number of cameras: ";
        if (std::cin >> num_cameras && num_cameras >= 1 && num_cameras <= 3) {
            break;
        }
        std::cin.clear();
        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        std::cout << "Invalid input\n";
    }

    // Path to exported model
    std::string model_path = "C:/Users/XXXX";

    // Load model 
    cv::dnn::Net net;
    try {
        net = cv::dnn::readNetFromONNX(model_path);
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_DEFAULT);
        //net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        std::cout << "Model loaded successfully from: " << model_path << "\n";
    } catch (const cv::Exception& e) {
        std::cerr << "Error loading model: " << e.what() << "\n";
        return -1;
    }

    //Initialize camera streams dynamically
    std::vector<std::pair<int, CameraStream*>> streams;
    for (int i = 0; i < num_cameras; ++i) {
        CameraStream* stream = new CameraStream(i);
        stream->start();
        streams.push_back({i, stream});
    }

    std::cout << "Starting stream with " << num_cameras << " camera(s). Press 'q' to exit.\n";

    //Main processing loop
    try {
        while (true) {
            for (auto& pair : streams) {
                int cam_idx = pair.first;
                CameraStream* stream = pair.second;

                cv::Mat frame;
                if (stream->read(frame)) {
                    // Preprocess frame for YOLO
                    cv::Mat blob;
                    cv::dnn::blobFromImage(frame, blob, 1.0 / 255.0, cv::Size(640, 640), cv::Scalar(0, 0, 0), true, false);
                    net.setInput(blob);
                    
                    // Forward pass
                    cv::Mat outputs = net.forward();
                    
                    // Display frame
                    cv::imshow("YOLO - Camera " + std::to_string(cam_idx), frame);
                }
            }

            // Break loop on pressing 'q'
            if (cv::waitKey(1) == 'q') {
                break;
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Exception in main loop: " << e.what() << "\n";
    }

    //Cleanup resources
    for (auto& pair : streams) {
        delete pair.second;
    }
    cv::destroyAllWindows();

    return 0;
}