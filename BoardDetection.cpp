#include "BoardDetection.h"
#include <numeric>

BoardExtractor::BoardExtractor() {
    // Initialization if needed
}

cv::Mat BoardExtractor::_orderPointsRotationProof(const cv::Mat& pts) {
    // pts is 4x2
    cv::Mat ordered = pts.clone();
    
    // 1. Compute Center
    float center_x = 0.0f;
    float center_y = 0.0f;
    for (int i = 0; i < 4; i++)
    {
        center_x += pts.at<float>(i,0);
        center_y += pts.at<float>(i,1);
    }
    
    // cv::Scalar center_scalar = cv::mean(pts);
    // cv::Point2f center((float)center_scalar[0], (float)center_scalar[1]);
    cv::Point2f center(center_x/4, center_y/4);

    // 2. Compute angles relative to center
    std::vector<int> indices(4);
    std::iota(indices.begin(), indices.end(), 0);
    
    std::vector<float> angles;
    for (int i = 0; i < 4; i++) {
        float dx = pts.at<float>(i, 0) - center.x;
        float dy = pts.at<float>(i, 1) - center.y;
        angles.push_back(std::atan2(dy, dx));
    }

    // 3. Sort indices by angle (Counter-Clockwise)
    std::sort(indices.begin(), indices.end(), [&](int a, int b) {
        return angles[a] < angles[b];
    });

    // Reorder based on angles
    cv::Mat sorted_by_angle(4, 2, CV_32F);
    for (int i = 0; i < 4; i++) {
        sorted_by_angle.at<float>(i, 0) = pts.at<float>(indices[i], 0);
        sorted_by_angle.at<float>(i, 1) = pts.at<float>(indices[i], 1);
    }

    // 4. Determine Top-Left (smallest x^2 + y^2)
    // In your Python: np.argmin((ordered**2).sum(axis=1))
    int topmost_index = 0;
    float min_dist_sq = 1e10;
    for (int i = 0; i < 4; i++) {
        float x = sorted_by_angle.at<float>(i, 0);
        float y = sorted_by_angle.at<float>(i, 1);
        float dist_sq = x*x + y*y;
        if (dist_sq < min_dist_sq) {
            min_dist_sq = dist_sq;
            topmost_index = i;
        }
    }

    // 5. Roll the array so topmost_index is at 0
    cv::Mat final_ordered(4, 2, CV_32F);
    for (int i = 0; i < 4; i++) {
        int src_idx = (i + topmost_index) % 4;
        final_ordered.at<float>(i, 0) = sorted_by_angle.at<float>(src_idx, 0);
        final_ordered.at<float>(i, 1) = sorted_by_angle.at<float>(src_idx, 1);
    }

    return final_ordered;
}

cv::Mat BoardExtractor::extractBoard(const cv::Mat& inputImg) {
    // 1. Detect corners (The function we built previously)
    // This already handles internal resizing and scaling back to original coordinates
    cv::Mat corners = detectChessboardCorners(inputImg);
    
    if (corners.empty()) {
        return cv::Mat();
    }
    corners = corners * 640/500;
    
    // 2. Apply rotation proof ordering
    cv::Mat orderedCorners = _orderPointsRotationProof(corners);

    return orderedCorners;
}

// In BoardDetection.cpp
std::pair<cv::Mat, cv::Mat> BoardExtractor::warp(const cv::Mat& img, const cv::Mat& quad, cv::Size target_size) {
    // Define the destination corners (standard square board)
    std::vector<cv::Point2f> dstPoints = {
        {0, 0}, 
        {(float)target_size.width, 0}, 
        {(float)target_size.width, (float)target_size.height}, 
        {0, (float)target_size.height}
    };
    
    // Extract source points from the 4x2 matrix
    std::vector<cv::Point2f> srcPoints;
    for(int i = 0; i < 4; i++) {
        srcPoints.push_back(cv::Point2f(quad.at<float>(i, 0), quad.at<float>(i, 1)));
    }

    // Calculate Transformation Matrix M
    cv::Mat M = cv::getPerspectiveTransform(srcPoints, dstPoints);
    
    cv::Mat warped;
    cv::warpPerspective(img, warped, M, target_size);
    
    return {warped, M};
}