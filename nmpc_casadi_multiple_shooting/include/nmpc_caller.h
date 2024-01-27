
#ifndef NMPC_CALLER_H
#define NMPC_CALLER_H

#include <casadi/casadi.hpp>
#include <casadi/core/generic_matrix.hpp>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <vector>
#include <yaml-cpp/yaml.h>

using namespace casadi;
using namespace std;

class nmpc_caller {
public:
  nmpc_caller();
  ~nmpc_caller();
  //定义求解器
  void set_nmpc_solver();
  //优化求解
  void opti_solution(Eigen::Matrix<float, 12, 1> current_states,
                     Eigen::VectorXf desired_states,
                     Eigen::Matrix<float, 12, 1> desired_controls,
                     Eigen::MatrixXf desired_gait, Eigen::MatrixXf _p_feet);
  //获取最优控制向量
  Eigen::VectorXf get_controls();
  //获取预测轨迹
  Eigen::VectorXf get_predict_trajectory();

  // 求向量的反对称矩阵
  SX rpy2rot(SX rpy);
  // 角速度转换
  SX omega2rpydot(SX rpy, SX omega);
  // 四阶龙格库塔法
  SX ode45(SX state_vars, SX control_vars, SX p_feet);
  // 一阶欧拉法
  SX Eular_method(SX state_vars, SX control_vars, SX p_feet);

  SX motion_function(SX state_vars, SX control_vars, SX p_feet);

private:
  SX weightQ;
  SX weightR;
  SX weightS;
  SX weightQ_t;

  float mass;
  float mu;
  SX g;
  SX inertia;
  SX inertia_inv;

  int f_num;

  int horizon; // horizon数
  float dtmpc; // mpc预测的时间步长

  Eigen::VectorXf predict_trajectory; //预测轨迹
  Eigen::VectorXf control_command;    //求解得到的控制量

  Function solver;     //求解器
  map<string, DM> res; //求解结果
  map<string, DM> arg; //求解参数
};

#endif // NMPC_CALLER_H