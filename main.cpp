#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <filesystem>

constexpr double VIDEO_SAVE_RESIZE_COEF = 0.5;
constexpr const char* input_video_path = "./data/advio-15/iphone/frames.mov";

struct SavedVideoParams {
    int frame_width;
    int frame_height;
    double fps;
};

void process_frame(cv::Mat& frame){
    // ToDo: add VO
}

struct SavedVideoParams get_params(const cv::VideoCapture& cap){ 
    struct SavedVideoParams params;
    params.frame_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH) * VIDEO_SAVE_RESIZE_COEF);
    params.frame_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT) * VIDEO_SAVE_RESIZE_COEF);
    params.fps = cap.get(cv::CAP_PROP_FPS);
    return params;
}

bool initialize_writers(const struct SavedVideoParams& video_params, cv::VideoWriter& writer){

    // writer for original video with marks
    writer.open(
        "./output/output.mp4", 
        cv::VideoWriter::fourcc('m', 'p', '4', 'v'), // MP4 codec
        video_params.fps, 
        cv::Size(video_params.frame_width, video_params.frame_height)
    );

    if (!writer.isOpened()) {
        std::cerr << "Error: Cannot open the video writer.\n" << std::endl;
        return false;
    }

    // if success return true
    return true;
}

void write_to_file(const cv::Mat& frame, const struct SavedVideoParams& video_params, cv::VideoWriter& writer, bool is_grey){
    cv::Mat frame_tmp;
    cv::resize(frame, frame_tmp, cv::Size(video_params.frame_width, video_params.frame_height));
    if (is_grey){
        cv::cvtColor(frame_tmp, frame_tmp, cv::COLOR_GRAY2BGR);
    }
    writer.write(frame_tmp);
}

int main() {
    std::filesystem::create_directory("./output");

    // Open the video file
    cv::VideoCapture cap(input_video_path);
    if (!cap.isOpened()) {
        std::cerr << "Error: Cannot open the video file." << std::endl;
        return -1;
    }

    // for saving video
    struct SavedVideoParams video_params = get_params(cap);
    cv::VideoWriter writer, writer_proc;
    bool writer_init_is_successful = initialize_writers(video_params, writer);
    if (!writer_init_is_successful){
        std::cerr << "Error: Writers init wasn't successful." << std::endl;
        return -1;
    }
    
    // Window to show the video
    cv::namedWindow("Video", cv::WINDOW_AUTOSIZE);

    cv::Mat frame;
    while (true) {
        // Read next frame
        cap >> frame;

        // Check if frame is empty (end of video)
        if (frame.empty())
            break;

        process_frame(frame);

        // Display the frame
        cv::imshow("Video with detection", frame);

        write_to_file(frame, video_params, writer, false);

        // Wait for 30ms and break if 'q' is pressed
        if (cv::waitKey(30) == 'q')
            break;
    }

    // Release resources
    cap.release();
    writer.release();
    cv::destroyAllWindows();
    return 0;
}
