#include "PieceCropper.h"
#include <iostream>

using namespace cv;
using namespace std;

// --- CameraMapper Implementation ---

Mat CameraMapper::getK(Size current_size, float f, Size original_size) {
    current_size.width = 3;
    float fx = f * (float)current_size.width / original_size.width;
    float fy = f * (float)current_size.height / original_size.height;
    // Python uses cx, cy as center of the CURRENT image
    Mat K = (Mat_<float>(3,3) << fx, 0, current_size.width/2.0f,
                                 0, fy, current_size.height/2.0f,
                                 0, 0, 1);
    return K;
}

void CameraMapper::getExtrinsics(const Mat& image_pts, const Mat& K, Mat& R, Mat& t) {
    // 1. Python uses [0,0], [1,0], [1,1], [0,1] for L=1.0
    vector<Point2f> world_pts = {{0,0}, {1,0}, {1,1}, {0,1}};
    
    vector<Point2f> img_pts_vec;
    for (int i = 0; i < 4; ++i) {
        img_pts_vec.push_back(image_pts.at<Point2f>(i, 0));
    }
    
    Mat H = findHomography(world_pts, img_pts_vec);
    H.convertTo(H, CV_32F);
    
    Mat Kinv = K.inv();
    Mat h1 = H.col(0);
    Mat h2 = H.col(1);
    Mat h3 = H.col(2);

    float lam = 1.0f / norm(Kinv * h1);
    Mat r1 = lam * (Kinv * h1);
    Mat r2 = lam * (Kinv * h2);
    Mat r3 = r1.cross(r2);

    Mat R_temp;
    hconcat(r1, r2, R_temp);
    hconcat(R_temp, r3, R_temp);

    SVD svd(R_temp);
    R = svd.u * svd.vt;
    if (determinant(R) < 0) R.col(2) *= -1;

    // IMPORTANT: Python logic calculates t as (-R) * cam_point later.
    // To match get_K_R from python, we use the h3 translation directly here
    // but we must be careful with the cam_point offset.
    t = lam * (Kinv * h3);
}

Mat CameraMapper::traceRay(const Mat& points_2d, const Mat& K, const Mat& R_ext, float Z_target) {
    Mat Kinv = K.inv();
    Mat R = R_ext.colRange(0, 3);
    Mat t = R_ext.col(3);
    
    // Camera position in World Space
    Mat cam_pos_world = -R.t() * t; 
    float cam_z = cam_pos_world.at<float>(2);

    Mat points_3d(points_2d.rows, 3, CV_32F);
    
    for(int i = 0; i < points_2d.rows; ++i) {
        Point2f pt_2d = points_2d.at<Point2f>(i, 0);
        Mat p_img = (Mat_<float>(3,1) << pt_2d.x, pt_2d.y, 1.0f);
        
        // Ray direction in World Space
        Mat dir_world = R.t() * (Kinv * p_img);
        
        // Distance to target Z plane
        // Equation: cam_pos.z + t_param * dir_world.z = Z_target
        float t_param = (Z_target - cam_z) / dir_world.at<float>(2);
        
        Mat p_3d = cam_pos_world + t_param * dir_world;
        
        points_3d.at<float>(i, 0) = p_3d.at<float>(0);
        points_3d.at<float>(i, 1) = p_3d.at<float>(1);
        points_3d.at<float>(i, 2) = Z_target;
    }
    return points_3d;
}

Mat CameraMapper::project3D(const Mat& points_3d, const Mat& K, const Mat& R_ext) {
    Mat res(points_3d.rows, 1, CV_32FC2);
    Mat R = R_ext.colRange(0, 3);
    Mat t = R_ext.col(3);

    for(int i = 0; i < points_3d.rows; ++i) {
        Mat p_w = (Mat_<float>(3,1) << points_3d.at<float>(i,0), 
                                       points_3d.at<float>(i,1), 
                                       points_3d.at<float>(i,2));
        
        // Transform: Camera = R*World + t
        Mat p_c = R * p_w + t;
        Mat p_2d = K * p_c;
        
        float z = p_2d.at<float>(2);
        res.at<Point2f>(i, 0) = Point2f(p_2d.at<float>(0)/z, p_2d.at<float>(1)/z);
    }
    return res;
}

// --- PieceCropper Implementation ---

PieceCropper::PieceCropper() {
    grid_original_ = Mat(81, 1, CV_32FC2);
    int idx = 0;
    for(int i = 0; i < 9; ++i) {
        for(int j = 0; j < 9; ++j) {
            grid_original_.at<Point2f>(idx++) = Point2f(j * 32.0f, i * 32.0f);
        }
    }
}

vector<Mat> PieceCropper::process(const Mat& img, const Mat& corners) {
    // Get camera intrinsics
    Mat K = CameraMapper::getK(img.size());
    
    // Get extrinsics from corners
    Mat R, t, R_ext;
    CameraMapper::getExtrinsics(corners, K, R, t);
    Mat cam_point = (Mat_<float>(3,1) << 0.0f, 0.0f, 0.5f);
    t = (-R) * cam_point;
    hconcat(R, t, R_ext);

    // Setup warp from corners to 256x256 space
    vector<Point2f> dst_pts = {{0,0}, {256,0}, {256,256}, {0,256}};
    vector<Point2f> src_pts_vec;
    for(int i = 0; i < 4; ++i) {
        src_pts_vec.push_back(corners.at<Point2f>(i, 0));
    }
    Mat M = getPerspectiveTransform(src_pts_vec, dst_pts);
    Mat Minv = M.inv();

    // Transform grid to image space (bottom grid)
    Mat grid_2d_normal;
    perspectiveTransform(grid_original_, grid_2d_normal, Minv);

    // Generate Top Grid via Ray Tracing
    // First lift points to 3D at Z=0
    Mat grid_3d = CameraMapper::traceRay(grid_2d_normal, K, R_ext, 0.0f);
    
    // Lift by 0.15 units (piece height)
    grid_3d.col(2) += 0.15f;
    
    // Project back to 2D
    Mat grid_top_mat = CameraMapper::project3D(grid_3d, K, R_ext);

    // Clip coordinates to image bounds
    int H = img.rows;
    int W = img.cols;
    
    for (int i = 0; i < grid_top_mat.rows; ++i) {
        Point2f& pt = grid_top_mat.at<Point2f>(i, 0);
        pt.x = max(0.0f, min(float(W - 1), pt.x));
        pt.y = max(0.0f, min(float(H - 1), pt.y));
    }
    
    for (int i = 0; i < grid_2d_normal.rows; ++i) {
        Point2f& pt = grid_2d_normal.at<Point2f>(i, 0);
        pt.x = max(0.0f, min(float(W - 1), pt.x));
        pt.y = max(0.0f, min(float(H - 1), pt.y));
    }

    // Reshape to 9x9 for extractWarpedSquares
    Mat gt = grid_top_mat.reshape(2, 9);
    Mat gb = grid_2d_normal.reshape(2, 9);

    return extractWarpedSquares(img, gt, gb, 64, 128);
}

std::vector<cv::Mat> extractWarpedSquares(
    const cv::Mat& image,
    const cv::Mat& grid_top,
    const cv::Mat& grid_bottom,
    int sq_width,
    int sq_height
) {
    if (image.empty() || image.type() != CV_8UC3) {
        throw invalid_argument("Image must be CV_8UC3");
    }

    if (grid_top.rows != 9 || grid_top.cols != 9 || grid_top.type() != CV_32FC2) {
        throw invalid_argument("grid_top must be 9x9 CV_32FC2");
    }

    if (grid_bottom.rows != 9 || grid_bottom.cols != 9 || grid_bottom.type() != CV_32FC2) {
        throw invalid_argument("grid_bottom must be 9x9 CV_32FC2");
    }

    vector<cv::Mat> squares;
    squares.reserve(64);

    Point2f dst_corners[4] = {
        {0.f, 0.f},
        {float(sq_width), 0.f},
        {float(sq_width), float(sq_height)},
        {0.f, float(sq_height)}
    };

    for (int r = 0; r < 8; ++r) {
        for (int c = 0; c < 8; ++c) {
            // --- Top edge (use top grid) ---
            Point2f tl_top = grid_top.at<Point2f>(r, c);
            Point2f tr_top = grid_top.at<Point2f>(r, c + 1);

            float x_min_top = min(tl_top.x, tr_top.x);
            float x_max_top = max(tl_top.x, tr_top.x);
            float y_min_top = min(tl_top.y, tr_top.y);

            // --- Bottom edge (use bottom grid) ---
            Point2f bl_bot = grid_bottom.at<Point2f>(r + 1, c);
            Point2f br_bot = grid_bottom.at<Point2f>(r + 1, c + 1);

            float x_min_bottom = min(bl_bot.x, br_bot.x);
            float x_max_bottom = max(bl_bot.x, br_bot.x);
            float y_max_bottom = max(bl_bot.y, br_bot.y);

            Point2f src_corners[4] = {
                {x_min_top,    y_min_top},      // Top-left
                {x_max_top,    y_min_top},      // Top-right
                {x_max_bottom, y_max_bottom},   // Bottom-right
                {x_min_bottom, y_max_bottom}    // Bottom-left
            };

            // --- Perspective transform ---
            Mat M = getPerspectiveTransform(src_corners, dst_corners);

            Mat warped;
            warpPerspective(image, warped, M, Size(sq_width, sq_height));

            // --- Draw debug markers (blue dots at bottom corners) ---
            Point2f bottom_corners[4] = {
                grid_bottom.at<Point2f>(r, c),
                grid_bottom.at<Point2f>(r, c + 1),
                grid_bottom.at<Point2f>(r + 1, c + 1),
                grid_bottom.at<Point2f>(r + 1, c)
            };

            vector<Point2f> warped_pts;
            perspectiveTransform(
                vector<Point2f>(bottom_corners, bottom_corners + 4),
                warped_pts,
                M
            );

            for (const auto& p : warped_pts) {
                int x0 = max(int(p.x) - 5, 0);
                int y0 = max(int(p.y) - 5, 0);
                int x1 = min(int(p.x) + 5, sq_width);
                int y1 = min(int(p.y) + 5, sq_height);

                for (int y = y0; y < y1; ++y) {
                    for (int x = x0; x < x1; ++x) {
                        warped.at<Vec3b>(y, x) = Vec3b(0, 0, 255); // Blue in BGR
                    }
                }
            }

            squares.push_back(warped);
        }
    }

    return squares;
}