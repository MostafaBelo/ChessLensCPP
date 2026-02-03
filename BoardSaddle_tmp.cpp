#include "BoardSaddle.h"
#include <set>

using namespace cv;
using namespace std;

namespace ChessboardDetection {

// --- Internal Helper Functions to match original logic ---

static Mat getIdentityGrid(int N) {
    Mat grid(N * N, 2, CV_32F);
    for (int y = 0; y < N; y++) {
        for (int x = 0; x < N; x++) {
            grid.at<float>(y * N + x, 0) = (float)x;
            grid.at<float>(y * N + x, 1) = (float)y;
        }
    }
    return grid;
}

static tuple<Mat, Mat, Mat> getInitChessGrid(const vector<Point>& quad) {
    Mat quadMat(4, 2, CV_32F);
    for (int i = 0; i < 4; i++) {
        quadMat.at<float>(i, 0) = (float)quad[i].x;
        quadMat.at<float>(i, 1) = (float)quad[i].y;
    }
    Mat quadA = (Mat_<float>(4, 2) << 0, 1, 1, 1, 1, 0, 0, 0);
    Mat M = getPerspectiveTransform(quadA, quadMat);
    return createChessGrid(M, 1);
}

// --- Implementation ---

double computeAngle(double a, double b, double c) {
    double k = (a * a + b * b - c * c) / (2 * a * b);
    return acos(max(-1.0, min(1.0, k))) * 180.0 / CV_PI;
}

bool isSquare(const vector<Point>& cnt, double eps) {
    if (cnt.size() != 4) return false;
    auto dist = [](Point p1, Point p2) { return sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2)); };
    double d0 = dist(cnt[0], cnt[1]), d1 = dist(cnt[1], cnt[2]);
    double d2 = dist(cnt[2], cnt[3]), d3 = dist(cnt[3], cnt[0]);
    double xa = dist(cnt[0], cnt[2]), xb = dist(cnt[1], cnt[3]);
    double ta = computeAngle(d3, d0, xb), tb = computeAngle(d0, d1, xa);
    double tc = computeAngle(d1, d2, xb), td = computeAngle(d2, d3, xa);
    bool good_angles = (ta > 40 && ta < 140) && (tb > 40 && tb < 140) && (tc > 40 && tc < 140) && (td > 40 && td < 140);
    return good_angles && (max(d0/d1, d1/d0) < eps) && abs(round(ta+tb+tc+td) - 360) < 5;
}

vector<Point> refineCorners(const vector<Point>& contour, const Mat& saddle, int ws) {
    vector<Point> new_contour = contour;
    for (size_t i = 0; i < contour.size(); i++) {
        int cc = contour[i].x, rr = contour[i].y;
        int rl = max(0, rr - ws), cl = max(0, cc - ws);
        int rh = min(saddle.rows - 1, rr + ws), ch = min(saddle.cols - 1, cc + ws);
        Mat window = saddle(Range(rl, rh + 1), Range(cl, ch + 1));
        double maxVal; Point maxLoc;
        minMaxLoc(window, nullptr, &maxVal, nullptr, &maxLoc);
        if (maxVal > 0) new_contour[i] = Point(cl + maxLoc.x, rl + maxLoc.y);
        else return {};
    }
    return new_contour;
}

Mat computeSaddle(const Mat& gray_img) {
    Mat img; gray_img.convertTo(img, CV_64F);
    Mat gx, gy, gxx, gyy, gxy;
    Sobel(img, gx, CV_64F, 1, 0); Sobel(img, gy, CV_64F, 0, 1);
    Sobel(gx, gxx, CV_64F, 1, 0); Sobel(gy, gyy, CV_64F, 0, 1);
    Sobel(gx, gxy, CV_64F, 0, 1);
    return gxx.mul(gyy) - gxy.mul(gxy);
}

void pruneSaddle(Mat& s, int max_pts) {
    double thresh = 128;
    while (countNonZero(s > 0) > max_pts) {
        thresh *= 2;
        s.setTo(0, s < thresh);
    }
}

Mat nonMaxSuppression(const Mat& img, int win) {
    Mat img_sup = Mat::zeros(img.size(), CV_64F);
    for (int y = 0; y < img.rows; ++y) {
        for (int x = 0; x < img.cols; ++x) {
            double val = img.at<double>(y, x);
            if (val <= 0) continue;
            int y0 = max(0, y - win), y1 = min(img.rows - 1, y + win);
            int x0 = max(0, x - win), x1 = min(img.cols - 1, x + win);
            double maxVal;
            minMaxLoc(img(Range(y0, y1 + 1), Range(x0, x1 + 1)), nullptr, &maxVal);
            if (val == maxVal) img_sup.at<double>(y, x) = val;
        }
    }
    return img_sup;
}

void extractContours(const Mat& edges, vector<vector<Point>>& contours, vector<Vec4i>& hierarchy) {
    Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(3, 3)), grad;
    morphologyEx(edges, grad, MORPH_GRADIENT, kernel);
    vector<vector<Point>> tmp;
    findContours(grad, tmp, hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE);
    contours.clear();
    for (const auto& c : tmp) {
        vector<Point> approx;
        approxPolyDP(c, approx, 0.04 * arcLength(c, true), true);
        contours.push_back(approx);
    }
}

void filterContours(const vector<vector<Point>>& c_all, const vector<Vec4i>& h_all, const Mat& saddle, 
                    vector<vector<Point>>& filtered, vector<Vec4i>& h_filt, double area_tol) {
    vector<pair<vector<Point>, double>> valid;
    for (size_t i = 0; i < c_all.size(); i++) {
        if (h_all[i][2] != -1 || c_all[i].size() != 4 || contourArea(c_all[i]) < 64) continue;
        vector<Point> updated = refineCorners(c_all[i], saddle, 4);
        if (!updated.empty()) valid.push_back({updated, contourArea(updated)});
    }
    if (valid.empty()) return;
    vector<double> areas;
    for (auto& p : valid) areas.push_back(p.second);
    sort(areas.begin(), areas.end());
    double median = areas[areas.size() / 2];
    for (auto& p : valid) {
        if (p.second >= area_tol * median && p.second <= 2.0 * median) filtered.push_back(p.first);
    }
}

tuple<Mat, Mat, Mat> createChessGrid(const Mat& M, int N) {
    Mat ideal = getIdentityGrid(2 + 2 * N);
    ideal -= Scalar(N, N);
    
    Mat grid(ideal.rows, 2, CV_32F);
    Mat M64; M.convertTo(M64, CV_64F);
    for (int i = 0; i < ideal.rows; i++) {
        double x = ideal.at<float>(i, 0), y = ideal.at<float>(i, 1);
        double w = M64.at<double>(2, 0) * x + M64.at<double>(2, 1) * y + M64.at<double>(2, 2);
        grid.at<float>(i, 0) = (float)((M64.at<double>(0, 0) * x + M64.at<double>(0, 1) * y + M64.at<double>(0, 2)) / w);
        grid.at<float>(i, 1) = (float)((M64.at<double>(1, 0) * x + M64.at<double>(1, 1) * y + M64.at<double>(1, 2)) / w);
    }
    return make_tuple(grid, ideal, M.clone());
}

tuple<Mat, Mat> findGridMatches(const Mat& grid, const vector<Point>& spts, double max_dist) {
    Mat res = grid.clone(), mask = Mat::zeros(grid.rows, 1, CV_8U);
    set<string> used;
    for (int i = 0; i < grid.rows; i++) {
        Point2f p(grid.at<float>(i, 0), grid.at<float>(i, 1));
        Point2f best_p; double min_d2 = 1e9;
        for (const auto& sp : spts) {
            double d2 = pow(sp.x - p.x, 2) + pow(sp.y - p.y, 2);
            if (d2 < min_d2) { min_d2 = d2; best_p = sp; }
        }
        string id = to_string((int)best_p.x) + "_" + to_string((int)best_p.y);
        if (sqrt(min_d2) < max_dist && used.find(id) == used.end()) {
            res.at<float>(i, 0) = best_p.x; res.at<float>(i, 1) = best_p.y;
            mask.at<uchar>(i, 0) = 1; used.insert(id);
        }
    }
    return make_tuple(res, mask);
}

Mat fitHomography(const Mat& ideal, const Mat& detected, const Mat& mask) {
    vector<Point2f> a, b;
    for (int i = 0; i < detected.rows; i++) {
        if (mask.at<uchar>(i, 0)) {
            a.push_back(Point2f(ideal.at<float>(i, 0), ideal.at<float>(i, 1)));
            b.push_back(Point2f(detected.at<float>(i, 0), detected.at<float>(i, 1)));
        }
    }
    return (a.size() < 4) ? Mat() : findHomography(a, b, RANSAC);
}

tuple<vector<int>, vector<int>> detectBestLines(const Mat& warped) {
    Mat blr, gx, gy; blur(warped, blr, Size(5, 5));
    Sobel(blr, gx, CV_64F, 1, 0); Sobel(blr, gy, CV_64F, 0, 1);
    
    auto score_axis = [](const Mat& g, bool horizontal) {
        Mat pos = g.clone(); pos.setTo(0, pos < 0);
        Mat neg = -g; neg.setTo(0, neg < 0);
        int n = horizontal ? g.rows : g.cols;
        vector<double> scores(n);
        for(int i=0; i<n; i++) {
            Mat line = horizontal ? g.row(i) : g.col(i);
            Mat lp = horizontal ? pos.row(i) : pos.col(i);
            Mat ln = horizontal ? neg.row(i) : neg.col(i);
            scores[i] = sum(lp)[0] * sum(ln)[0];
        }
        vector<double> combo(8, 0);
        for(int off=1; off<=8; off++) 
            for(int j=0; j<7; j++) {
                int idx = (off + j + 1) * 32;
                if(idx < n) combo[off-1] += scores[idx];
            }
        int best_off = (int)(max_element(combo.begin(), combo.end()) - combo.begin() + 1);
        vector<int> res(7);
        for(int j=0; j<7; j++) res[j] = (best_off + j + 1) * 32;
        return res;
    };
    return make_tuple(score_axis(gx, false), score_axis(gy, true));
}

} // namespace

// --- Main API Implementation ---

Mat detectChessboardCorners(Mat img, const ChessboardDetectionConfig& config) {
    if (img.channels() > 1) cvtColor(img, img, COLOR_RGB2GRAY);
    cv::imwrite("unscalled.png", img);
    if (img.cols > config.max_image_size || img.rows > config.max_image_size) {
        double s = min((double)config.max_image_size/img.cols, (double)config.max_image_size/img.rows);
        resize(img, img, Size(), s, s, INTER_LINEAR);
    }
    cv::imwrite("scalled.png", img);

    
    // Store scale for coordinate reconstruction
    double s = 1.0;
    if (img.cols > config.max_image_size || img.rows > config.max_image_size) {
        s = std::min((double)config.max_image_size / img.cols, (double)config.max_image_size / img.rows);
        resize(img, img, cv::Size(), s, s, cv::INTER_LINEAR);
    }
    
    Mat blurred; blur(img, blurred, Size(3, 3));
    Mat saddle = ChessboardDetection::computeSaddle(blurred);
    saddle = -saddle; saddle.setTo(0, saddle < 0);
    ChessboardDetection::pruneSaddle(saddle);
    Mat s2 = ChessboardDetection::nonMaxSuppression(saddle);
    s2.setTo(0, s2 < 100000);

    vector<Point> spts;
    for (int y = 0; y < s2.rows; y++)
        for (int x = 0; x < s2.cols; x++)
            if (s2.at<double>(y, x) != 0) spts.push_back(Point(x, y));

    Mat edges; Canny(img, edges, config.canny_low, config.canny_high);
    vector<vector<Point>> c_all, cnts; vector<Vec4i> h_all, h_filt;
    ChessboardDetection::extractContours(edges, c_all, h_all);
    ChessboardDetection::filterContours(c_all, h_all, saddle, cnts, h_filt, 0.25);

    Mat best_M, best_grid, best_mask; int max_good = 0;
    for (const auto& cnt : cnts) {
        Mat g_curr, g_ideal, M; tie(g_curr, g_ideal, M) = ChessboardDetection::getInitChessGrid(cnt);
        int n_good = 0; Mat g_next, mask;
        for (int i = 0; i < 7; i++) {
            tie(g_curr, g_ideal, std::ignore) = ChessboardDetection::createChessGrid(M, i + 1);
            tie(g_next, mask) = ChessboardDetection::findGridMatches(g_curr, spts, config.max_px_dist);
            n_good = countNonZero(mask);
            if (n_good < 4) { M.release(); break; }
            M = ChessboardDetection::fitHomography(g_ideal, g_next, mask);
            if (M.empty() || abs(M.at<double>(0,0)/M.at<double>(1,1)) > 15) { M.release(); break; }
        }
        if (!M.empty() && n_good > max_good) {
            max_good = n_good; best_M = M.clone(); best_grid = g_next.clone(); best_mask = mask.clone();
        }
        if (max_good > config.max_pts_needed) break;
    }

    if (max_good <= config.min_pts_needed) return Mat();

    Mat final_ideal(256, 2, CV_32F);
    for(int y=0; y<16; y++) for(int x=0; x<16; x++) {
        final_ideal.at<float>(y*16+x, 0) = (float)(x - 7 + 8) * 32.0f;
        final_ideal.at<float>(y*16+x, 1) = (float)(y - 7 + 8) * 32.0f;
    }
    best_M = ChessboardDetection::fitHomography(final_ideal, best_grid, best_mask);
    if (best_M.empty()) return Mat();

    Mat warped; warpPerspective(img, warped, best_M, Size(17*32, 17*32), WARP_INVERSE_MAP);
    vector<int> bx, by; tie(bx, by) = ChessboardDetection::detectBestLines(warped);

    int d = bx[1] - bx[0];
    vector<Point2f> corners_warp = {
        {(float)bx[0]-d, (float)by[0]-d}, {(float)bx.back()+d, (float)by[0]-d},
        {(float)bx.back()+d, (float)by.back()+d}, {(float)bx[0]-d, (float)by.back()+d}
    };
    Mat res; perspectiveTransform(Mat(corners_warp).reshape(2, 4), res, best_M);

    if (res.empty()) return cv::Mat();

    // SCALE BACK TO ORIGINAL COORDINATES
    res = res / s;

    return res.reshape(1, 4); // Returns 4x2 CV_32F
}