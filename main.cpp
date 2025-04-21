#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <filesystem>

constexpr double VIDEO_SAVE_RESIZE_COEF = 0.5;
// constexpr const char* INPUT_VIDEO_PATH = "./data/advio-01/iphone/frames.mov";
constexpr const char* INPUT_VIDEO_PATH = "./data/road.mp4";
constexpr int GUESSED_FOCAL_LENGTH = 700;
constexpr bool SHOW_KEYPOINTS = true;

static cv::Mat pose = cv::Mat::eye(4, 4, CV_64F);
static int inlier_count = 0;
static int matches_count = 0;

struct SavedVideoParams {
    int frame_width;
    int frame_height;
    double fps;
};

void process_frame(cv::Mat& frame1, cv::Mat& frame2, std::vector<cv::Point2f>& keypoints_to_display){

    // keypoints
    cv::Ptr<cv::ORB> orb = cv::ORB::create();
    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    cv::Mat descriptors1, descriptors2;

    cv::Mat gray1, gray2;
    cv::cvtColor(frame1, gray1, cv::COLOR_BGR2GRAY);
    cv::cvtColor(frame2, gray2, cv::COLOR_BGR2GRAY);
    orb->detectAndCompute(gray1, cv::noArray(), keypoints1, descriptors1);
    orb->detectAndCompute(gray2, cv::noArray(), keypoints2, descriptors2);

    // matching
    std::vector<std::vector<cv::DMatch>> matches;
    cv::BFMatcher matcher(cv::NORM_HAMMING);
    matcher.knnMatch(descriptors1, descriptors2, matches, 2);

    // filter only good matches
    std::vector<cv::Point2f> pts1, pts2;
    int n_matches = 0;
    for (auto &match : matches) {
        if (match.size() == 2 && match[0].distance < 0.75f * match[1].distance){
            pts1.push_back(keypoints1[match[0].queryIdx].pt);
            pts2.push_back(keypoints2[match[0].trainIdx].pt);
            n_matches++;
        }
    }
    matches_count = n_matches;

    cv::Mat inlier_mask;
    cv::Mat fundamental_matrix = cv::findFundamentalMat(pts1, pts2, cv::FM_RANSAC, 2.0, 0.99, inlier_mask);

    // filtering inliers
    std::vector<cv::Point2f> inlier_pts1, inlier_pts2;
    for (int i = 0; i < inlier_mask.rows; ++i) {
        if (inlier_mask.at<uchar>(i)) {
            inlier_pts1.push_back(pts1[i]);
            inlier_pts2.push_back(pts2[i]);
        }
    }
    inlier_count = cv::countNonZero(inlier_mask);

    // rough estimate of camera intrinsics
    cv::Mat camera_intrinsics = (cv::Mat_<double>(3, 3) << 
        GUESSED_FOCAL_LENGTH, 0, frame1.cols / 2,
        0, GUESSED_FOCAL_LENGTH, frame1.rows / 2,
        0, 0, 1);


    cv::Mat essential_martix = camera_intrinsics.t() * fundamental_matrix * camera_intrinsics;

    cv::Mat rotation_estimate, translation_direction;
    cv::recoverPose(essential_martix, pts1, pts2, camera_intrinsics, rotation_estimate, translation_direction);
    // std::cout << rotation_estimate << std::endl;
    // std::cout << translation_direction << std::endl;

    cv::Mat relative_motion = cv::Mat::eye(4, 4, CV_64F);
    rotation_estimate.copyTo(relative_motion(cv::Rect(0, 0, 3, 3)));
    translation_direction.copyTo(relative_motion(cv::Rect(3, 0, 1, 3)));

    // Update global pose
    pose = pose * relative_motion;
    keypoints_to_display = inlier_pts2;

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

void visualize_trajectory(cv::Mat& traj){

    // Extract camera translation (world position)
    double x = pose.at<double>(0, 3);
    double z = pose.at<double>(2, 3);  // We use X-Z plane for simplicity

    // Convert to image coordinates
    int draw_x = static_cast<int>(x * 5 + traj.cols / 2); // scale & center
    int draw_z = static_cast<int>(z * 5 + traj.rows / 2);

    cv::circle(traj, cv::Point(draw_x, draw_z), 2, cv::Scalar(0, 255, 0), -1);
    cv::imshow("Trajectory", traj);
}

void display_frame(cv::Mat& frame, std::vector<cv::Point2f>& keypoints_to_display){
    cv::Mat frame_copy = frame.clone();
    std::string info = "N inliers: " + std::to_string(inlier_count) + " among " + std::to_string(matches_count) + " matches";
    cv::putText(frame_copy, info, cv::Point(30, 30), 
        cv::FONT_HERSHEY_SIMPLEX , 1.2, 
        cv::Scalar(0, 0, 255), 1.5, cv::LINE_AA);

    // display keypoints
    if (SHOW_KEYPOINTS){
        for (auto point : keypoints_to_display){
            cv::circle(frame_copy, point, 1, cv::Scalar(0, 0, 255), 2);
        }
    }

    cv::imshow("Video", frame_copy);
}

int main() {
    std::filesystem::create_directory("./output");

    // Open the video file
    cv::VideoCapture cap(INPUT_VIDEO_PATH);
    if (!cap.isOpened()) {
        std::cerr << "Error: Cannot open the video file." << std::endl;
        return -1;
    }

    // for saving video
    struct SavedVideoParams video_params = get_params(cap);
    cv::VideoWriter writer;
    bool writer_init_is_successful = initialize_writers(video_params, writer);
    if (!writer_init_is_successful){
        std::cerr << "Error: Writers init wasn't successful." << std::endl;
        return -1;
    }
    
    cv::Mat traj = cv::Mat::zeros(600, 600, CV_8UC3);

    // Windows
    cv::namedWindow("Video", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Trajectory", cv::WINDOW_AUTOSIZE);

    cv::Mat frame1, frame2;
    cap >> frame1;
    if (frame1.empty()){
        std::cerr << "Error: No frame." << std::endl;
        return -1;
    }
        
    while (true) {
        // Read next frame
        cap >> frame2;

        // Check if frame is empty (end of video)
        if (frame2.empty())
            break;

        std::vector<cv::Point2f> keypoints_to_display;
        process_frame(frame1, frame2, keypoints_to_display);
        display_frame(frame2, keypoints_to_display);
        write_to_file(frame2, video_params, writer, false);

        frame1 = frame2;
        visualize_trajectory(traj);

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
