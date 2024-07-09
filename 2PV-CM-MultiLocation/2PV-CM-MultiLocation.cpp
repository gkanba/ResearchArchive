#include <iostream>
#include <cstdint>
#include <vector>
#include <set>
#include <string>
#include <chrono>
#include <fstream>

#include <omp.h>
#include <eigen3/Eigen/Dense>
#include <nlohmann/json.hpp>

#define EIGEN_DONT_PARALLELIZE

constexpr double pi = 3.14159265358979323846;

static const Eigen::IOFormat IOFMT(Eigen::FullPrecision, 0, ", ", "\n", "", "");

Eigen::VectorXd RHS_T(const int32_t N, const Eigen::VectorXd circulation, const Eigen::VectorXd X)
{
    Eigen::VectorXd rhs = Eigen::VectorXd::Zero(2 * N);
    for(int m = 0; m < N; m++){
        double_t sum_x = 0.0;
        double_t sum_y = 0.0;
        for(int k = 0; k < N; k++){
            if(k != m){
                double_t r = std::pow((X(m) - X(k)), 2) + std::pow((X(m + N) - X(k + N)), 2);
                sum_x += circulation(k) * ((X(m + N) - X(k + N))) / r;
                sum_y += circulation(k) * ((X(  m  ) - X(  k  ))) / r;
            }
        }
        rhs(  m  ) = - (1.0 / (2.0 * pi)) * sum_x;
        rhs(m + N) =   (1.0 / (2.0 * pi)) * sum_y;
    }
    return rhs;
}

Eigen::VectorXd RHS_A(const int32_t N, Eigen::VectorXd circulation, const Eigen::VectorXd X, const Eigen::VectorXd X_T, const double_t mu, const std::set<int32_t> n_list)
{
    Eigen::VectorXd rhs = Eigen::VectorXd::Zero(2 * N);
    for(int m = 0; m < N; m++){
        double_t sum_x = 0.0;
        double_t sum_y = 0.0;
        for(int k = 0; k < N; k++){
            if(k != m){
                double_t r = std::pow((X(m) - X(k)), 2) + std::pow((X(m + N) - X(k + N)), 2);
                sum_x += circulation(k) * ((X(m + N) - X (k + N))) / r;
                sum_y += circulation(k) * ((X(  m  ) - X (  k  ))) / r;
            }
        }
        // For first Vortex which is observable
        if(n_list.find(m) != n_list.end()){
            rhs(  m  ) = - (1.0 / (2.0 * pi)) * sum_x + mu * (X_T(  m  ) - X(  m  ));
            rhs(m + N) =   (1.0 / (2.0 * pi)) * sum_y + mu * (X_T(m + N) - X(m + N));
        }
        // For second to last Vortex which is not observable
        else{
            rhs(  m  ) = - (1.0 / (2.0 * pi)) * sum_x;
            rhs(m + N) =   (1.0 / (2.0 * pi)) * sum_y;
        }
    }
    return rhs;
}

void SaveMatrixObj(const char* obj_name, std::string file_name, Eigen::MatrixXd obj, Eigen::IOFormat format){
    std::cout << "[IO] " << "Trying to save matrix object " << obj_name << " as " << file_name << std::endl;
    std::ofstream stream(file_name);
    if(stream.is_open()){
        stream << obj.format(format) << std::endl;
        std::cout << "[IO] " << "Saved matrix object " << obj_name << " as " << file_name << std::endl;
    }
    else{
        std::cout << "[IO] " << "Unable to open file " << file_name << std::endl;
        std::cout << "[IO] " << "Failed to save matrix object " << obj_name << " as " << file_name << std::endl;
    }
}

template <class Rep, std::intmax_t num, std::intmax_t denom>
auto chronoBurst(std::chrono::duration<Rep, std::ratio<num, denom>> d)
{
    const auto hrs  = std::chrono::duration_cast<std::chrono::hours>(d);
    const auto mins = std::chrono::duration_cast<std::chrono::minutes>(d - hrs);
    const auto secs = std::chrono::duration_cast<std::chrono::seconds>(d - hrs - mins);
    const auto ms   = std::chrono::duration_cast<std::chrono::milliseconds>(d - hrs - mins - secs);

    return std::make_tuple(hrs, mins, secs, ms);
}

namespace settings{
    struct JsonConfig{
        std::string RootDir;
        int32_t Threads;

        double_t Delta_t;
        double_t Max_t;
        double_t Tau_t;

        int32_t VortexCount;
        std::vector<double_t> Circulations;
        std::vector<double_t> InitCenters;

        double_t Mu;
        std::set<int32_t> NudgeList;

        std::vector<double_t> Range_x;
        std::vector<double_t> Range_y;

        int32_t Div_x;
        int32_t Div_y;

    };

    void to_json(nlohmann::json& j, const JsonConfig& jc) {
        j = nlohmann::json{
            {"RootDir",         jc.RootDir},
            {"Threads",         jc.Threads},
            {"Delta_t",         jc.Delta_t},
            {"Max_t",           jc.Max_t},
            {"Tau_t",           jc.Tau_t},
            {"VortexCount",     jc.VortexCount},
            {"Circulations",    jc.Circulations},
            {"InitCenters",     jc.InitCenters},
            {"Mu",              jc.Mu},
            {"NudgeList",       jc.NudgeList},
            {"Range_x",         jc.Range_x},
            {"Range_y",         jc.Range_y},
            {"Div_x",           jc.Div_x},
            {"Div_y",           jc.Div_y}
        };
    }

    void from_json(const nlohmann::json& j, JsonConfig& jc){
        j.at("RootDir").get_to(jc.RootDir);
        j.at("Threads").get_to(jc.Threads);
        j.at("Delta_t").get_to(jc.Delta_t);
        j.at("Max_t").get_to(jc.Max_t);
        j.at("Tau_t").get_to(jc.Tau_t);
        j.at("VortexCount").get_to(jc.VortexCount);
        j.at("Circulations").get_to(jc.Circulations);
        j.at("InitCenters").get_to(jc.InitCenters);
        j.at("Mu").get_to(jc.Mu);
        j.at("NudgeList").get_to(jc.NudgeList);
        j.at("Range_x").get_to(jc.Range_x);
        j.at("Range_y").get_to(jc.Range_y);
        j.at("Div_x").get_to(jc.Div_x);
        j.at("Div_y").get_to(jc.Div_y);
    }

};

int32_t main(int32_t argc, char** argv){

    // Load Json Configuration
    std::ifstream ifs("config.json");
    nlohmann::json jsonobj = nlohmann::json::parse(ifs);
    ifs.close();

    // Output Json Configuration
    std::cout << "========================================================" << std::endl;
    std::cout << "Loading config.json || Initialized with parameters below" << std::endl;
    std::cout << jsonobj.dump(4) << std::endl;
    
    // Deserialize Json Object
    auto config = jsonobj.template get<settings::JsonConfig>();

    // Create Experiment Directory
    std::cout << "========================================================" << std::endl;
    std::cout << "Creating Experiment Directory || " << config.RootDir << std::endl;
    bool isOverrided = std::filesystem::create_directory(config.RootDir);
    if(!isOverrided){
        std::cout << "Directory of the same name detected, overrided the directory : " << config.RootDir << std::endl;
    }
    else{
        std::cout << "Created new directory" << config.RootDir << std::endl;
    }

    std::ofstream ofs(config.RootDir + "/log.json");
    ofs << jsonobj;
    ofs.close();

    // Check Threads Avaliability
    int32_t num_threads = config.Threads;
    std::cout << "========================================================" << std::endl;
    std::cout << "Checking Threads availability || Threads count set in the config : " << config.Threads << std::endl;
    if(omp_get_max_threads() < num_threads){
        std::cout << "Threads count exceeded || Max Threads count : " << omp_get_max_threads() << config.RootDir << std::endl;
        omp_set_num_threads(omp_get_max_threads());
    }
    else{
        std::cout << config.Threads << " Threads are available || Max Threads count : " << omp_get_max_threads() << std::endl;
        omp_set_num_threads(num_threads);
    }

    // Load Parameters
    double_t delta_t    = config.Delta_t;
    double_t max_t      = config.Max_t;
    double_t tau_t      = config.Tau_t;

    int32_t max_step    = std::ceil(max_t / delta_t);
    int32_t tau_step    = std::ceil(tau_t / delta_t);

    int32_t N           = config.VortexCount;
    double_t mu         = config.Mu;
    std::set<int32_t> nlist = config.NudgeList;

    int32_t div_x       = config.Div_x;
    int32_t div_y       = config.Div_y;
    double_t delta_x    = (config.Range_x.at(1) - config.Range_x.at(0)) / div_x;
    double_t delta_y    = (config.Range_y.at(1) - config.Range_y.at(0)) / div_y;

    // Every x,y coordinates for the deisgnated row, column
    Eigen::VectorXd pos_x = Eigen::VectorXd::Zero(div_x + 1);
    for(int32_t i = 0; i < div_x + 1; i++){
        pos_x(i) = config.Range_x.at(0) + i * delta_x;
    }
    Eigen::VectorXd pos_y = Eigen::VectorXd::Zero(div_y + 1);
    for(int32_t i = 0; i < div_y + 1; i++){
        pos_y(i) = config.Range_y.at(1) - i * delta_y;
    }

    // Circulation Initialization
    Eigen::VectorXd circulation = Eigen::VectorXd::Zero(N);
    for(int32_t i = 0; i < config.Circulations.size(); i++){
        circulation(i) = config.Circulations.at(i);
    }

    // Center : True
    // centers_T(0, :) : Centers at t = 0  
    Eigen::MatrixXd centers_T = Eigen::MatrixXd::Zero(N * 2, max_step + 1);
    for(int32_t i = 0; i < config.InitCenters.size(); i++){
        centers_T(i, 0) = config.InitCenters.at(i);
    }

    // Error Mean Matrix Initialization
    Eigen::MatrixXd error_mean = Eigen::MatrixXd::Zero(div_y + 1, div_x + 1);
    Eigen::MatrixXd error_diff = Eigen::MatrixXd::Zero(div_y + 1, div_x + 1);

    // === Details output ===
    std::cout << "========================================================" << std::endl;
    std::cout << "=== Started All Procedures ===" << std::endl;
    std::chrono::system_clock::time_point s_all = std::chrono::system_clock::now();

    // === Integrate Centers_T ===
    std::cout << "========================================================" << std::endl;
    std::cout << "Started Procedure : Integrating Truth Centers" << std::endl;
    std::chrono::system_clock::time_point s_int_cT = std::chrono::system_clock::now();
    for(int32_t s = 0; s < max_step; s++){
        Eigen::VectorXd cv = centers_T.col(s);
        Eigen::VectorXd k1 = RHS_T(N, circulation, cv);
        Eigen::VectorXd k2 = RHS_T(N, circulation, cv + 0.5 * delta_t * k1);
        Eigen::VectorXd k3 = RHS_T(N, circulation, cv + 0.5 * delta_t * k2);
        Eigen::VectorXd k4 = RHS_T(N, circulation, cv +       delta_t * k3);
        centers_T.col(s + 1) = centers_T.col(s) + (delta_t / 6.0) * (k1 + 2 * (k2 + k3) + k4);
    }

    std::chrono::system_clock::time_point e_int_cT = std::chrono::system_clock::now();
    const auto [hrs_int_cT, mins_int_cT, secs_int_cT, ms_int_cT] = chronoBurst(e_int_cT - s_int_cT);
    std::cout << "Completed Procedure : Integrating Truth Centers (" << max_step << " steps done)" << std::endl;
    std::cout << "Total Time Elapsed Here : " << hrs_int_cT.count() << "h" << mins_int_cT.count() << "m" << secs_int_cT.count() << "s and " << ms_int_cT.count() << "ms" << std::endl;

    // === Integrate Centers_A ===
    std::cout << "========================================================" << std::endl;
    std::cout << "Started All Procedure : Integrating Assimilated Centers" << std::endl;
    int32_t c_int_cA = 0;
    std::chrono::system_clock::time_point s_int_cA = std::chrono::system_clock::now();
    #pragma omp parallel for
    for(int32_t ix = 0; ix < div_x + 1; ix++){
        #pragma omp parallel for
        for(int32_t iy = 0; iy < div_y + 1; iy++){
            // Log output

            #pragma omp critical
            {
                if(c_int_cA % 100 == 0 && c_int_cA != 0){
                    std::chrono::system_clock::time_point m_int_cA = std::chrono::system_clock::now();
                    const auto [hrs, mins, secs, ms] = chronoBurst(m_int_cA - s_int_cA);
                    std::cout << "Completed : c = " << c_int_cA <<std::endl;
                    std::cout << "Time Elapsed : " << hrs.count() << "h" << mins.count() << "m" << secs.count() << "s and " << ms.count() << "ms" << std::endl;
                }
            }
            
            // Get Current Coordinate for examination
            Eigen::MatrixXd l_centers_A = Eigen::MatrixXd::Zero(N * 2, max_step + 1);
            for(int32_t i = 0; i < config.InitCenters.size(); i++){
                l_centers_A(i, 0) = config.InitCenters.at(i);
            }
            l_centers_A(N - 1, 0)       = pos_x(ix);
            l_centers_A(2 * N - 1, 0)   = pos_y(iy);

            for(int32_t s = 0; s < max_step; s++){
                Eigen::VectorXd cv_A = l_centers_A.col(s);
                Eigen::VectorXd cv_T = centers_T.col(s);
                Eigen::VectorXd k1 = RHS_A(N, circulation, cv_A,                      cv_T, mu, nlist);
                Eigen::VectorXd k2 = RHS_A(N, circulation, cv_A + 0.5 * delta_t * k1, cv_T, mu, nlist);
                Eigen::VectorXd k3 = RHS_A(N, circulation, cv_A + 0.5 * delta_t * k2, cv_T, mu, nlist);
                Eigen::VectorXd k4 = RHS_A(N, circulation, cv_A +       delta_t * k3, cv_T, mu, nlist);
                l_centers_A.col(s + 1) = l_centers_A.col(s) + (delta_t / 6.0) * (k1 + 2 * (k2 + k3) + k4);
            }
            
            double_t mean_error_sum = 0.0;
            double_t max_error = 0.0;
            double_t min_error = 10000000.0;

            // Store l2 norm for every time step
            Eigen::VectorXd errors_X = Eigen::VectorXd::Zero(max_step + 1);
            for(int32_t s = 0; s < max_step; s++){
                errors_X(s) = (centers_T.col(s) - l_centers_A.col(s)).norm();
            }

            for(int32_t s = max_step - tau_step + 1; s <= max_step; s++){
                if(errors_X(s) > max_error) max_error = errors_X(s);
                if(errors_X(s) < min_error) min_error = errors_X(s);
                mean_error_sum += errors_X(s);
            }

            #pragma omp critical
            {
                error_mean(iy, ix) = mean_error_sum / tau_step;
                error_diff(iy, ix) = max_error - min_error;
                c_int_cA++;
            }
        }
    }
    std::chrono::system_clock::time_point e_int_cA = std::chrono::system_clock::now();
    const auto [hrs_int_cA, mins_int_cA, secs_int_cA, ms_int_cA] = chronoBurst(e_int_cA - s_int_cA);
    std::cout << "Completed Procedure : Integrating Assimilated Centers (" << c_int_cA << " points done)" << std::endl;
    std::cout << "Total Time Elapsed Here : " << hrs_int_cA.count() << "h" << mins_int_cA.count() << "m" << secs_int_cA.count() << "s and " << ms_int_cA.count() << "ms" << std::endl;

    // IO : Save Matrices
    std::cout << "========================================================" << std::endl;
    std::cout << "IO Procedure : Saving Data Matrices" << std::endl;
    SaveMatrixObj("centers_T", config.RootDir + "/Centers_T.csv", centers_T, IOFMT);
    //SaveMatrixObj("centers_A", "Centers_A.csv", centers_A, IOFMT);
    SaveMatrixObj("error_mean", config.RootDir + "/ErrorMeanMatrix.csv", error_mean, IOFMT);
    SaveMatrixObj("error_diff", config.RootDir + "/ErrorDiffMatrix.csv", error_diff, IOFMT);

    std::chrono::system_clock::time_point e_all = std::chrono::system_clock::now();
    std::cout << "=== Completed All Procedures === " << std::endl;
    const auto [hrs_all, mins_all, secs_all, ms_all] = chronoBurst(e_all - s_all);
    std::cout << "Total Time Elapsed : " << hrs_all.count() << "h" << mins_all.count() << "m" << secs_all.count() << "s and " << ms_all.count() << "ms" << std::endl;
    std::cout << "========================================================" << std::endl;
}
