#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <filesystem>

// constexpr const char* INPUT_VIDEO_PATH = "./data/advio-01/iphone/frames.mov";
constexpr const char* INPUT_VIDEO_PATH = "./data/road.mp4";
constexpr int GUESSED_FOCAL_LENGTH = 700;
constexpr bool SHOW_KEYPOINTS = true;
constexpr int TRAJECTORY_VIS_SCALE = 1;

struct VOState {
    cv::Mat pose = cv::Mat::eye(4,4,CV_64F);
    int inlier_count = 0, matches_count = 0;
};

void calculate_keypoints(cv::Ptr<cv::ORB> orb, const cv::Mat& frame, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors){
    cv::Mat gray;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    orb->detectAndCompute(gray, cv::noArray(), keypoints, descriptors);
}

void calculate_pose(
        struct VOState& vo_state,
        const std::vector<cv::KeyPoint>& keypoints1,
        const std::vector<cv::KeyPoint>& keypoints2,
        const cv::Mat& descriptors1,
        const cv::Mat& descriptors2,
        const cv::Mat& camera_intrinsics,
        std::vector<cv::Point2f>& keypoints_to_display
        ){
    
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
    vo_state.matches_count = n_matches;

    cv::Mat inlier_mask;
    cv::Mat essential_matrix = cv::findEssentialMat(
        pts1, pts2, camera_intrinsics,
        cv::RANSAC, 0.99, 1.0, inlier_mask
    );

    // filtering inliers
    std::vector<cv::Point2f> inlier_pts1, inlier_pts2;
    for (int i = 0; i < inlier_mask.rows; ++i) {
        if (inlier_mask.at<uchar>(i)) {
            inlier_pts1.push_back(pts1[i]);
            inlier_pts2.push_back(pts2[i]);
        }
    }
    vo_state.inlier_count = cv::countNonZero(inlier_mask);

    cv::Mat rotation_estimate, translation_direction;
    cv::recoverPose(essential_matrix, pts1, pts2, camera_intrinsics, rotation_estimate, translation_direction);
    // std::cout << rotation_estimate << std::endl;
    // std::cout << translation_direction << std::endl;

    cv::Mat relative_motion = cv::Mat::eye(4, 4, CV_64F);
    rotation_estimate.copyTo(relative_motion(cv::Rect(0, 0, 3, 3)));
    translation_direction.copyTo(relative_motion(cv::Rect(3, 0, 1, 3)));

    // Update global pose
    vo_state.pose = vo_state.pose * relative_motion;
    keypoints_to_display = inlier_pts2;

}

void visualize_trajectory(const struct VOState& vo_state, cv::Mat& traj){

    // Extract camera translation (world position)
    double x = vo_state.pose.at<double>(0, 3);
    double z = vo_state.pose.at<double>(2, 3);  // We use X-Z plane for simplicity

    // Convert to image coordinates
    int draw_x = static_cast<int>(x * TRAJECTORY_VIS_SCALE + traj.cols / 2); // scale & center
    int draw_z = static_cast<int>(z * TRAJECTORY_VIS_SCALE + traj.rows / 2);

    cv::circle(traj, cv::Point(draw_x, draw_z), 2, cv::Scalar(0, 255, 0), -1);
    cv::imshow("Trajectory", traj);
}

void display_frame(const struct VOState& vo_state, cv::Mat& frame, std::vector<cv::Point2f>& keypoints_to_display){
    cv::Mat frame_copy = frame.clone();
    std::string info = "N inliers: " + std::to_string(vo_state.inlier_count) + " among " + std::to_string(vo_state.matches_count) + " matches";
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

    struct VOState vo_state;

    // Open the video file
    cv::VideoCapture cap(INPUT_VIDEO_PATH);
    if (!cap.isOpened()) {
        std::cerr << "Error: Cannot open the video file." << std::endl;
        return -1;
    }
    
    cv::Mat traj = cv::Mat::zeros(600, 600, CV_8UC3);

    // keypoint detector
    cv::Ptr<cv::ORB> orb = cv::ORB::create();

    // Windows
    cv::namedWindow("Video", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Trajectory", cv::WINDOW_AUTOSIZE);

    cv::Mat frame1, frame2;
    cap >> frame1;
    if (frame1.empty()){
        std::cerr << "Error: No frame." << std::endl;
        return -1;
    }

    // rough estimate of camera intrinsics
    cv::Mat camera_intrinsics = (cv::Mat_<double>(3, 3) << 
        GUESSED_FOCAL_LENGTH, 0, frame1.cols / 2,
        0, GUESSED_FOCAL_LENGTH, frame1.rows / 2,
        0, 0, 1);
    
    std::vector<cv::KeyPoint> keypoints1;
    cv::Mat descriptors1;
    calculate_keypoints(orb, frame1, keypoints1, descriptors1);

    while (true) {
        // Read next frame
        cap >> frame2;

        // Check if frame is empty (end of video)
        if (frame2.empty())
            break;

        std::vector<cv::Point2f> keypoints_to_display;
        std::vector<cv::KeyPoint> keypoints2;
        cv::Mat descriptors2;
        calculate_keypoints(orb, frame2, keypoints2, descriptors2);

        calculate_pose(vo_state, keypoints1, keypoints2, descriptors1, descriptors2, camera_intrinsics, keypoints_to_display);
        display_frame(vo_state, frame2, keypoints_to_display);
        visualize_trajectory(vo_state, traj);

        frame1 = frame2;
        keypoints1 = keypoints2;
        descriptors1 = descriptors2;

        // Wait for 30ms and break if 'q' is pressed
        if (cv::waitKey(30) == 'q')
            break;
    }

    // Release resources
    cap.release();
    cv::destroyAllWindows();
    return 0;
}
