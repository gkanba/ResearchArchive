#include <iostream>
#include <cstdint>
#include <vector>
#include <set>
#include <string>
#include <chrono>
#include <fstream>

#include <omp.h>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Core>
#include <nlohmann/json.hpp>

#define EIGEN_DONT_PARALLELIZE

constexpr double pi = 3.14159265358979323846;
static const Eigen::IOFormat IOFMT(Eigen::FullPrecision, 0, ", ", "\n", "", "");

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

Eigen::VectorXd RHS_T(const int32_t N, const Eigen::VectorXd circulation, const Eigen::VectorXd X)
{
    Eigen::VectorXd rhs = Eigen::VectorXd::Zero(2 * N);
    for(int m = 0; m < N; m++){
        double_t sum_x = 0.0;
        double_t sum_y = 0.0;
        for(int k = 0; k < N; k++){
            if(k != m){
                double_t rs = std::pow((X(m) - X(k)), 2) + std::pow((X(m + N) - X(k + N)), 2);
                sum_x += circulation(k) * ((X(m + N) - X(k + N))) / rs;
                sum_y += circulation(k) * ((X(  m  ) - X(  k  ))) / rs;
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
                double_t rs = std::pow((X(m) - X(k)), 2) + std::pow((X(m + N) - X(k + N)), 2);
                sum_x += circulation(k) * ((X(m + N) - X (k + N))) / rs;
                sum_y += circulation(k) * ((X(  m  ) - X (  k  ))) / rs;
            }
        }
        // For Vortices which is observable
        if(n_list.find(m) != n_list.end()){
            rhs(  m  ) = - (1.0 / (2.0 * pi)) * sum_x + mu * (X_T(  m  ) - X(  m  ));
            rhs(m + N) =   (1.0 / (2.0 * pi)) * sum_y + mu * (X_T(m + N) - X(m + N));
        }
        // For Vortices which is NOT observable
        else{
            rhs(  m  ) = - (1.0 / (2.0 * pi)) * sum_x;
            rhs(m + N) =   (1.0 / (2.0 * pi)) * sum_y;
        }
    }
    return rhs;
}

namespace settings{
    struct JsonConfig{
        std::string RootDir;                    // Root Directory                                   :
        int32_t Threads;                        // Number of thread for OpenMP Parallelization      :
        bool IsDebug;                           // Output Log                                       :

        double_t Delta_t;                       // Delta T of Experiment                            :
        double_t Max_t;                         // Max   T of Experiment                            :
        double_t Tau_t;                         // Tau   T of Experiment                            :

        int32_t VortexCount;                    // Number of Vortex                                 : Always 2 in this experiment
        std::vector<double_t> Circulations;     // Every Circulation        (Size = VortexCount)    : Gamma_s will be updated later
        std::vector<double_t> InitCenters_T;    // Every Initial Centers T  (Size = VortexCout * 2) : 
        std::vector<double_t> InitCenters_A;    // Every Initial Centers A  (Size = VortexCout * 2) : 

        double_t Mu;                            // Nudging Gain                                     : Mu will be updated later
        double_t Eta;                           // Max Threshould error adopted                     : 
        std::set<int32_t> NudgeList;            // Set of Nudged Vortex                             : Always 1 only in this experiment

        std::vector<double_t> Range_Mu;         // Range of Mu Iteration                            :
        std::vector<double_t> Range_GS;         // Range of Gamma_s Iteration                       :
        int32_t Div_Mu;                         // Division Count of Mu Iteration                   :
        int32_t Div_GS;                         // Division Count of Gamma_s Iteration              :
    };

    void to_json(nlohmann::json& j, const JsonConfig& jc) {
        j = nlohmann::json{
            {"RootDir",         jc.RootDir},
            {"Threads",         jc.Threads},
            {"IsDebug",         jc.IsDebug},

            {"Delta_t",         jc.Delta_t},
            {"Max_t",           jc.Max_t},
            {"Tau_t",           jc.Tau_t},

            {"VortexCount",     jc.VortexCount},
            {"Circulations",    jc.Circulations},
            {"InitCenters_T",   jc.InitCenters_T},
            {"InitCenters_A",   jc.InitCenters_A},

            {"Mu",              jc.Mu},
            {"Eta",             jc.Eta},
            {"NudgeList",       jc.NudgeList},

            {"Range_Mu",        jc.Range_Mu},
            {"Range_GS",        jc.Range_GS},
            {"Div_Mu",          jc.Div_Mu},
            {"Div_GS",          jc.Div_GS}
        };
    }

    void from_json(const nlohmann::json& j, JsonConfig& jc){
        j.at("RootDir").get_to(jc.RootDir);
        j.at("Threads").get_to(jc.Threads);
        j.at("IsDebug").get_to(jc.IsDebug);

        j.at("Delta_t").get_to(jc.Delta_t);
        j.at("Max_t").get_to(jc.Max_t);
        j.at("Tau_t").get_to(jc.Tau_t);

        j.at("VortexCount").get_to(jc.VortexCount);
        j.at("Circulations").get_to(jc.Circulations);
        j.at("InitCenters_T").get_to(jc.InitCenters_T);
        j.at("InitCenters_A").get_to(jc.InitCenters_A);

        j.at("Mu").get_to(jc.Mu);
        j.at("Eta").get_to(jc.Eta);
        j.at("NudgeList").get_to(jc.NudgeList);

        j.at("Range_Mu").get_to(jc.Range_Mu);
        j.at("Range_GS").get_to(jc.Range_GS);
        j.at("Div_Mu").get_to(jc.Div_Mu);
        j.at("Div_GS").get_to(jc.Div_GS);
    }

};

int32_t main(int32_t argc, char** argv){

    // Eigen::initParallel();

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
        std::cout << "Created new directory : " << config.RootDir << std::endl;
    }

    // Save Json Object as log.json
    std::ofstream ofs(config.RootDir + "/log.json");
    ofs << jsonobj;
    ofs.close();

    // Load Parameters
    double_t delta_t        = config.Delta_t;
    double_t max_t          = config.Max_t;
    double_t tau_t          = config.Tau_t;

    int32_t max_step        = std::ceil(max_t / delta_t);
    int32_t tau_step        = std::ceil(tau_t / delta_t);

    int32_t N               = config.VortexCount;
    double_t mu             = config.Mu;
    double_t eta            = config.Eta;
    std::set<int32_t> nlist = config.NudgeList;

    int32_t div_mu      = config.Div_Mu;
    int32_t div_gs      = config.Div_GS;

    double_t delta_mu   = (config.Range_Mu.at(1) - config.Range_Mu.at(0)) / div_mu;
    double_t delta_gs   = (config.Range_GS.at(1) - config.Range_GS.at(0)) / div_gs;

    // Every Mu,GS, coordinates for the designated experiment
    Eigen::VectorXd pos_gs = Eigen::VectorXd::Zero(div_gs + 1);
    for(int32_t i = 0; i < div_gs + 1; i++){
        pos_gs(i) = config.Range_GS.at(0) + i * delta_gs;
    }
    Eigen::VectorXd pos_mu = Eigen::VectorXd::Zero(div_mu + 1);
    for(int32_t i = 0; i < div_mu + 1; i++){
        pos_mu(i) = config.Range_Mu.at(1) - i * delta_mu;
    }

    Eigen::MatrixXd mean_error_mat = Eigen::MatrixXd::Zero(div_mu + 1, div_gs + 1);
    Eigen::MatrixXd diff_error_mat = Eigen::MatrixXd::Zero(div_mu + 1, div_gs + 1);

    std::chrono::system_clock::time_point s_all = std::chrono::system_clock::now();

    #pragma omp parallel for
    for(int32_t imu = 0; imu < div_mu + 1; imu++){
        #pragma omp parallel for
        for(int32_t igs = 0; igs < div_gs + 1; igs++){

            double_t mu_tmp = pos_mu(imu);
            double_t gs_tmp = pos_gs(igs);

            // Circulation Initialization
            Eigen::VectorXd circulation = Eigen::VectorXd::Zero(N);
            
            // Insert Gamma_l Gamma_s
            circulation(0) = config.Circulations.at(0);
            circulation(1) = gs_tmp;

            // Truth Center Initialization
            Eigen::MatrixXd centers_T = Eigen::MatrixXd::Zero(N * 2, max_step + 1);
            for(int32_t i = 0; i < config.InitCenters_T.size(); i++){
                centers_T(i, 0) = config.InitCenters_T.at(i);
            }

            // Integrating Truth Center
            for(int32_t s = 0; s < max_step; s++){
                Eigen::VectorXd cv = centers_T.col(s);
                Eigen::VectorXd k1 = RHS_T(N, circulation, cv);
                Eigen::VectorXd k2 = RHS_T(N, circulation, cv + 0.5 * delta_t * k1);
                Eigen::VectorXd k3 = RHS_T(N, circulation, cv + 0.5 * delta_t * k2);
                Eigen::VectorXd k4 = RHS_T(N, circulation, cv +       delta_t * k3);
                centers_T.col(s + 1) = centers_T.col(s) + (delta_t / 6.0) * (k1 + 2 * (k2 + k3) + k4);
            }

            Eigen::MatrixXd centers_A = Eigen::MatrixXd::Zero(N * 2, max_step + 1);
            for(int32_t i = 0; i < config.InitCenters_A.size(); i++){
                centers_A(i, 0) = config.InitCenters_A.at(i);
            }
            // Integrate Assimilated Center
            for(int32_t s = 0; s < max_step; s++){
                Eigen::VectorXd cv_A = centers_A.col(s);
                Eigen::VectorXd cv_T = centers_T.col(s);
                Eigen::VectorXd k1 = RHS_A(N, circulation, cv_A,                      cv_T, mu_tmp, nlist);
                Eigen::VectorXd k2 = RHS_A(N, circulation, cv_A + 0.5 * delta_t * k1, cv_T, mu_tmp, nlist);
                Eigen::VectorXd k3 = RHS_A(N, circulation, cv_A + 0.5 * delta_t * k2, cv_T, mu_tmp, nlist);
                Eigen::VectorXd k4 = RHS_A(N, circulation, cv_A +       delta_t * k3, cv_T, mu_tmp, nlist);
                centers_A.col(s + 1) = centers_A.col(s) + (delta_t / 6.0) * (k1 + 2 * (k2 + k3) + k4);
            }

            // Store l2 norm for every time step
            Eigen::VectorXd errors_X = Eigen::VectorXd::Zero(max_step + 1);
            for(int32_t s = 0; s < max_step; s++){
                errors_X(s) = (centers_T.col(s) - centers_A.col(s)).norm();
            }

            double_t mean_error_sum = 0.0;
            double_t max_error = 0.0;
            double_t min_error = 10000000.0;
            for(int32_t s = max_step - tau_step + 1; s <= max_step; s++){
                if(errors_X(s) > max_error) max_error = errors_X(s);
                if(errors_X(s) < min_error) min_error = errors_X(s);
                mean_error_sum += errors_X(s);
            }
            double_t mean_error = mean_error_sum / tau_step;
            double_t diff_error = max_error - min_error;

            #pragma omp critical
            {
                mean_error_mat(imu, igs) = mean_error;
                diff_error_mat(imu, igs) = diff_error;
            }
        }
    }

    // IO : Save Matrices
    std::cout << "========================================================" << std::endl;
    std::cout << "IO Procedure : Saving Data Matrices" << std::endl;
    SaveMatrixObj("mean_error_mat", config.RootDir + "/MeanErrorMuGs.csv", mean_error_mat, IOFMT);
    SaveMatrixObj("diff_error_mat", config.RootDir + "/DiffErrorMuGs.csv", diff_error_mat, IOFMT);

    std::chrono::system_clock::time_point e_all = std::chrono::system_clock::now();
    std::cout << "=== Completed All Procedures === " << std::endl;
    const auto [hrs_all, mins_all, secs_all, ms_all] = chronoBurst(e_all - s_all);
    std::cout << "Total Time Elapsed : " << hrs_all.count() << "h" << mins_all.count() << "m" << secs_all.count() << "s and " << ms_all.count() << "ms" << std::endl;
    std::cout << "========================================================" << std::endl;

}