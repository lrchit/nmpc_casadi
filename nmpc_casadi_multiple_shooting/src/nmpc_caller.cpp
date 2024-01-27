
#include <nmpc_caller.h>

nmpc_caller::nmpc_caller() {

  //惩罚矩阵
  weightQ = SX::eye(12);
  weightR = SX::eye(12);
  weightS = SX::eye(12);
  weightQ_t = SX::eye(12);

  // 读取参数
  YAML::Node config = YAML::LoadFile("../config/nmpc_config.yaml");

  weightQ(0, 0) = config["WeightQ"]["Kpcom"]["x"].as<float>();
  weightQ(1, 1) = config["WeightQ"]["Kpcom"]["y"].as<float>();
  weightQ(2, 2) = config["WeightQ"]["Kpcom"]["z"].as<float>();
  weightQ(3, 3) = config["WeightQ"]["Kdcom"]["x"].as<float>();
  weightQ(4, 4) = config["WeightQ"]["Kdcom"]["y"].as<float>();
  weightQ(5, 5) = config["WeightQ"]["Kdcom"]["z"].as<float>();
  weightQ(6, 6) = config["WeightQ"]["Kpbase"]["roll"].as<float>();
  weightQ(7, 7) = config["WeightQ"]["Kpbase"]["pitch"].as<float>();
  weightQ(8, 8) = config["WeightQ"]["Kpbase"]["yaw"].as<float>();
  weightQ(9, 9) = config["WeightQ"]["Kdbase"]["roll"].as<float>();
  weightQ(10, 10) = config["WeightQ"]["Kdbase"]["pitch"].as<float>();
  weightQ(11, 11) = config["WeightQ"]["Kdbase"]["yaw"].as<float>();

  weightR(0, 0) = config["WeightR"]["x"].as<float>();
  weightR(1, 1) = config["WeightR"]["y"].as<float>();
  weightR(2, 2) = config["WeightR"]["z"].as<float>();
  weightR(3, 3) = config["WeightR"]["x"].as<float>();
  weightR(4, 4) = config["WeightR"]["y"].as<float>();
  weightR(5, 5) = config["WeightR"]["z"].as<float>();
  weightR(6, 6) = config["WeightR"]["x"].as<float>();
  weightR(7, 7) = config["WeightR"]["y"].as<float>();
  weightR(8, 8) = config["WeightR"]["z"].as<float>();
  weightR(9, 9) = config["WeightR"]["x"].as<float>();
  weightR(10, 10) = config["WeightR"]["y"].as<float>();
  weightR(11, 11) = config["WeightR"]["z"].as<float>();

  weightS(0, 0) = config["WeightS"]["x"].as<float>();
  weightS(1, 1) = config["WeightS"]["y"].as<float>();
  weightS(2, 2) = config["WeightS"]["z"].as<float>();
  weightS(3, 3) = config["WeightS"]["x"].as<float>();
  weightS(4, 4) = config["WeightS"]["y"].as<float>();
  weightS(5, 5) = config["WeightS"]["z"].as<float>();
  weightS(6, 6) = config["WeightS"]["x"].as<float>();
  weightS(7, 7) = config["WeightS"]["y"].as<float>();
  weightS(8, 8) = config["WeightS"]["z"].as<float>();
  weightS(9, 9) = config["WeightS"]["x"].as<float>();
  weightS(10, 10) = config["WeightS"]["y"].as<float>();
  weightS(11, 11) = config["WeightS"]["z"].as<float>();

  float alpha = config["alpha"].as<float>();
  weightQ_t = alpha * weightQ;

  mass = config["mass"].as<float>();
  inertia = SX::eye(3);
  inertia(0, 0) = config["inertia"]["ixx"].as<float>();
  inertia(0, 1) = config["inertia"]["ixy"].as<float>();
  inertia(0, 2) = config["inertia"]["ixz"].as<float>();
  inertia(1, 0) = config["inertia"]["ixy"].as<float>();
  inertia(1, 1) = config["inertia"]["iyy"].as<float>();
  inertia(1, 2) = config["inertia"]["iyz"].as<float>();
  inertia(2, 0) = config["inertia"]["ixz"].as<float>();
  inertia(2, 1) = config["inertia"]["iyz"].as<float>();
  inertia(2, 2) = config["inertia"]["izz"].as<float>();
  inertia_inv = SX::inv(inertia);

  horizon = config["horizon"].as<float>();
  dtmpc = config["dt"].as<float>() *
          config["iterations_between_segment"].as<float>();

  mu = config["mu"].as<float>();

  g = SX(3, 1);
  g(2) = 9.81;

  predict_trajectory.setZero(horizon * 12);
  control_command.setZero(horizon * 12);
}

nmpc_caller::~nmpc_caller() {}

//创建求解器
void nmpc_caller::set_nmpc_solver() {

  // Variables，状态和控制输入为变量
  SX x = SX::sym("x", 24 * horizon);

  // Parameters，含有参考控制输入，当前轨迹，参考轨迹，足端位置
  SX p = SX::sym("p", 12 * horizon + 12 * (horizon + 1) + 12);
  f_num = 12 * horizon;

  // 代价函数
  SX f = SX::sym("cost_fun");
  f = 0;
  // 1. 状态损失，末端损失权重更大
  for (int k = 0; k < horizon - 1; k++) {
    SX states_err = x(Slice(f_num + 12 * k, f_num + 12 * k + 12)) -
                    p(Slice(f_num + 12 * (k + 1), f_num + 12 * (k + 1) + 12));
    f = f + SX::mtimes({states_err.T(), weightQ, states_err});
  }
  SX states_err =
      x(Slice(f_num + 12 * (horizon - 1), f_num + 12 * (horizon - 1) + 12)) -
      p(Slice(f_num + 12 * horizon, f_num + 12 * horizon + 12));
  f = f + SX::mtimes({states_err.T(), weightQ_t, states_err});

  // 2. 控制输入的正则化
  for (int k = 0; k < horizon; k++) {
    SX controls_val = x(Slice(12 * k, 12 * k + 12));
    f = f + SX::mtimes({controls_val.T(), weightR, controls_val});
  }

  // 3. 控制输入的变化量
  SX delta_controls_val = x(Slice(0, 12)) - p(Slice(0, 12));
  f = f + SX::mtimes({delta_controls_val.T(), weightS, delta_controls_val});
  for (int k = 1; k < horizon - 1; k++) {
    SX delta_controls_val = x(Slice(12 * (k + 1), 12 * (k + 1) + 12)) -
                            x(Slice(12 * k, 12 * k + 12));
    f = f + SX::mtimes({delta_controls_val.T(), weightS, delta_controls_val});
  }

  // 1. 状态空间约束，应该用龙格库塔法离散化
  // 1.1 当前状态与优化的状态起始点满足约束
  SX constraints = SX::vertcat(
      {x(Slice(f_num, f_num + 12)) -
       Eular_method(p(Slice(f_num, f_num + 12)), x(Slice(0, 12)),
                    p(Slice(24 * horizon + 12, 24 * horizon + 24)))});

  // 1.2 中间所有步的优化状态满足约束
  for (int k = 0; k < horizon - 1; k++) {
    // cout << constraints.size() << endl;
    constraints = SX::vertcat(
        {constraints,
         x(Slice(f_num + 12 * (k + 1), f_num + 12 * (k + 1) + 12)) -
             Eular_method(x(Slice(f_num + 12 * k, f_num + 12 * k + 12)),
                          x(Slice(12 * (k + 1), 12 * (k + 1) + 12)),
                          p(Slice(24 * horizon + 12, 24 * horizon + 24)))});
  }

  // 2. 摩擦金字塔约束
  for (int k = 0; k < horizon; k++) {
    for (int leg = 0; leg < 4; leg++) {
      constraints = SX::vertcat(
          {constraints, x(12 * k + 0 + leg * 3) - mu * x(12 * k + 2 + leg * 3),
           x(12 * k + 1 + leg * 3) - mu * x(12 * k + 2 + leg * 3),
           x(12 * k + 0 + leg * 3) + mu * x(12 * k + 2 + leg * 3),
           x(12 * k + 1 + leg * 3) + mu * x(12 * k + 2 + leg * 3),
           x(12 * k + 2 + leg * 3)});
    }
  }

  //构建求解器
  SXDict nlp_prob = {{"x", x}, {"p", p}, {"f", f}, {"g", constraints}};

  string solver_name = "ipopt";
  Dict nlp_opts;
  nlp_opts["expand"] = true;
  nlp_opts["ipopt.max_iter"] = horizon * (12 + 12 + 12) * 5;
  nlp_opts["ipopt.linear_solver"] = "ma27";
  nlp_opts["ipopt.print_level"] = 0;
  nlp_opts["print_time"] = 1;
  nlp_opts["ipopt.acceptable_obj_change_tol"] = 1e-3;
  nlp_opts["ipopt.acceptable_tol"] = 1e-3;
  nlp_opts["ipopt.tol"] = 1e-3;
  nlp_opts["ipopt.nlp_scaling_method"] = "gradient-based";
  nlp_opts["ipopt.constr_viol_tol"] = 1e-3;
  nlp_opts["ipopt.fixed_variable_treatment"] = "relax_bounds";

  solver = nlpsol("nlpsol", solver_name, nlp_prob, nlp_opts);

  //求解参数设置
  // Initial guess and bounds for the optimization variablese
  vector<double> lbx;
  vector<double> ubx;

  for (int i = 0; i < horizon; i++) {
    for (int j = 0; j < 24; j++) {
      lbx.push_back(-1e6);
      ubx.push_back(1e6);
    }
  }

  arg["lbx"] = lbx;
  arg["ubx"] = ubx;
}

SX nmpc_caller::motion_function(SX state_vars, SX control_vars, SX p_feet) {

  SX rotation_mat = rpy2rot(state_vars(Slice(6, 9, 1)));

  SX tau = cross(p_feet(Slice(0, 3)) - state_vars(Slice(0, 3)),
                 control_vars(Slice(0, 3)), -1) +
           cross(p_feet(Slice(3, 6)) - state_vars(Slice(0, 3)),
                 control_vars(Slice(3, 6)), -1) +
           cross(p_feet(Slice(6, 9)) - state_vars(Slice(0, 3)),
                 control_vars(Slice(6, 9)), -1) +
           cross(p_feet(Slice(9, 12)) - state_vars(Slice(0, 3)),
                 control_vars(Slice(9, 12)), -1);

  SX rhs = SX::vertcat(
      {state_vars(Slice(3, 6)),
       (control_vars(Slice(0, 3)) + control_vars(Slice(3, 6)) +
        control_vars(Slice(6, 9)) + control_vars(Slice(9, 12))) /
               mass -
           g,
       omega2rpydot(state_vars(Slice(6, 9)), state_vars(Slice(9, 12))),
       SX::mtimes(
           {inertia_inv,
            (SX::mtimes({rotation_mat.T(), tau}) -
             cross(state_vars(Slice(9, 12)),
                   SX::mtimes({inertia, state_vars(Slice(9, 12))}), -1))})});
  return rhs;
}

void nmpc_caller::opti_solution(Eigen::Matrix<float, 12, 1> current_states,
                                Eigen::VectorXf desired_states,
                                Eigen::Matrix<float, 12, 1> desired_controls,
                                Eigen::MatrixXf desired_gait,
                                Eigen::MatrixXf p_feet) {

  //求解参数设置
  // Initial guess and bounds for the optimization variablese
  vector<double> x0;
  for (int i = 0; i < horizon; i++) {
    for (int j = 0; j < 12; j++) {
      x0.push_back(desired_controls[j]);
    }
  }
  for (int i = 0; i < horizon; i++) {
    for (int j = 0; j < 12; j++) {
      x0.push_back(desired_states[j]);
    }
  }

  // Nonlinear bounds
  vector<double> lbg;
  vector<double> ubg;
  for (int i = 0; i < horizon; i++) {
    for (int j = 0; j < 12; j++) {
      lbg.push_back(0);
      ubg.push_back(0);
    }
  }
  for (int i = 0; i < horizon; i++) {
    for (int leg = 0; leg < 4; leg++) {
      lbg.push_back(-1e6);
      lbg.push_back(-1e6);
      lbg.push_back(0);
      lbg.push_back(0);
      lbg.push_back(0);
      ubg.push_back(0);
      ubg.push_back(0);
      ubg.push_back(1e6);
      ubg.push_back(1e6);
      ubg.push_back(200 * desired_gait(i, leg));
    }
  }

  // Original parameter values
  // 前12*horizon个为参考输入
  // 第（12*horizon+1）～（12*horizon+12）个为当前状态
  // 第（12*horizon+12+1）～（12*horizon+12+12*horizon）个为参考状态
  // 最后12个为足端与地面接触点
  vector<double> p0;
  for (int i = 0; i < horizon; i++) {
    for (int j = 0; j < 12; j++) {
      p0.push_back(desired_controls[j]);
    }
  }
  for (int j = 0; j < 12; j++) {
    p0.push_back(current_states[j]);
  }
  for (int i = 0; i < horizon; i++) {
    for (int j = 0; j < 12; j++) {
      p0.push_back(desired_states[j]);
    }
  }
  // 接触点
  for (int leg = 0; leg < 4; leg++) {
    for (int i = 0; i < 3; i++) {
      p0.push_back(p_feet(i, leg));
    }
  }

  arg["lbg"] = lbg;
  arg["ubg"] = ubg;
  arg["x0"] = x0;
  arg["p"] = p0;
  res = solver(arg);

  // cout << "Objective: " << res.at("f") << endl;
  // cout << "Optimal solution for p = \n" << arg.at("p") << ":" << endl;
  // cout << "Primal solution: \n" << res.at("x") << endl;
  // cout << "Dual solution (x): " << res.at("lam_x") << endl;
  // cout << "Dual solution (g): " << res.at("lam_g") << endl;

  vector<float> res_all(res["x"]);
  for (int i = 0; i < horizon; i++) {
    for (int j = 0; j < 12; j++) {
      predict_trajectory(i * 12 + j) = res_all[f_num + i * 12 + j];
      control_command(i * 12 + j) = res_all[i * 12 + j + 12];
    }
  }
}

Eigen::VectorXf nmpc_caller::get_controls() { return control_command; }

Eigen::VectorXf nmpc_caller::get_predict_trajectory() {
  return predict_trajectory;
}

// rpy转为旋转矩阵
SX nmpc_caller::rpy2rot(SX rpy) {
  SX Rx = SX::eye(3);
  SX Ry = SX::eye(3);
  SX Rz = SX::eye(3);
  SX R = SX::eye(3);

  Rz(0, 0) = cos(rpy(2));
  Rz(0, 1) = -sin(rpy(2));
  Rz(1, 0) = sin(rpy(2));
  Rz(1, 1) = cos(rpy(2));

  Ry(0, 0) = cos(rpy(1));
  Ry(0, 2) = sin(rpy(1));
  Ry(2, 0) = -sin(rpy(1));
  Ry(2, 2) = cos(rpy(1));

  Rx(1, 1) = cos(rpy(0));
  Rx(1, 2) = -sin(rpy(0));
  Rx(2, 1) = sin(rpy(0));
  Rx(2, 2) = cos(rpy(0));

  R = SX::mtimes({Rz, Ry, Rx});

  return R;
}

// 求向量的反对称矩阵
SX nmpc_caller::omega2rpydot(SX rpy, SX omega) {

  SX rpy_dot = SX(3, 1);
  SX trans_mat = SX::eye(3);

  // trans_mat(0, 0) = cos(rpy(1)) / cos(rpy(2));
  // trans_mat(0, 1) = sin(rpy(2)) / cos(rpy(1));
  // trans_mat(0, 2) = 0;
  // trans_mat(1, 0) = -sin(rpy(2));
  // trans_mat(1, 1) = cos(rpy(2));
  // trans_mat(1, 2) = 0;
  // trans_mat(2, 0) = cos(rpy(2)) * tan(rpy(1));
  // trans_mat(2, 1) = sin(rpy(2)) * tan(rpy(1));
  // trans_mat(2, 2) = 1;

  trans_mat(0, 0) = 1;
  trans_mat(0, 1) = sin(rpy(0)) * tan(rpy(1));
  trans_mat(0, 2) = cos(rpy(0)) * tan(rpy(1));
  trans_mat(1, 0) = 0;
  trans_mat(1, 1) = cos(rpy(0));
  trans_mat(1, 2) = sin(rpy(0));
  trans_mat(2, 0) = 0;
  trans_mat(2, 1) = sin(rpy(0)) / cos(rpy(1));
  trans_mat(2, 2) = cos(rpy(0)) / cos(rpy(1));

  rpy_dot = SX::mtimes({trans_mat, omega});

  return rpy_dot;
}

// 四阶龙格库塔法
SX nmpc_caller::ode45(SX state_vars, SX control_vars, SX p_feet) {

  SX k1 = dtmpc * motion_function(state_vars, control_vars, p_feet);
  SX k2 = dtmpc * motion_function(state_vars + k1 / 2, control_vars, p_feet);
  SX k3 = dtmpc * motion_function(state_vars + k2 / 2, control_vars, p_feet);
  SX k4 = dtmpc * motion_function(state_vars + k3, control_vars, p_feet);

  return state_vars + (k1 + 2 * k2 + 2 * k3 + k4) / 6;
}

// 欧拉法法
SX nmpc_caller::Eular_method(SX state_vars, SX control_vars, SX p_feet) {

  return dtmpc * motion_function(state_vars, control_vars, p_feet) + state_vars;
}

int main(int argc, char *argv[]) {

  //预测长度，对yaml文件写入
  const int horizon = 20;
  YAML::Node config = YAML::LoadFile("../config/nmpc_config.yaml");
  ofstream fout("../config/nmpc_config.yaml");
  config["horizon"] = horizon;
  fout << config;
  fout.close();

  // 足底位置
  Eigen::MatrixXf p_feet;
  p_feet.setZero(3, 4);
  p_feet << 0.4, 0.4, -0.4, -0.4, -0.2, 0.2, 0.2, -0.2, -0.5, -0.5, -0.5, -0.5;
  // cout << "p_feet\n" << p_feet << endl;

  float vel_z = 0.;

  // 步态信息
  Eigen::MatrixXf desired_gait;
  desired_gait.setOnes(horizon, 4);

  // 当前状态，参考状态序列，参考控制序列
  Eigen::VectorXf current_states, current_controls, desired_states,
      desired_controls;
  current_states.setZero(12);
  current_states << 0, 0, 0, 0, 0, vel_z, 0, 0, 0, 0, 0, 0;
  current_controls.setZero(12);
  desired_states.setZero(12 * horizon);
  desired_controls.setZero(12);
  desired_controls << 0, 0, 22, 0, 0, 22, 0, 0, 22, 0, 0, 22;

  // 生成nmpc类
  nmpc_caller nmpc_controller;
  nmpc_controller.set_nmpc_solver();

  // 调用优化
  double iteration_steps = 100;
  for (int i = 0; i < iteration_steps + 1; i++) {
    for (int j = 0; j < horizon; j++) {
      desired_states.block(j * 12, 0, 12, 1) << 0, 0,
          vel_z * 0.04 * (j + 1 + i), 0, 0, vel_z, 0, 0, 0, 0, 0, 0;
    }
    nmpc_controller.opti_solution(current_states, desired_states,
                                  desired_controls, desired_gait, p_feet);
    //获取最优控制向量
    Eigen::VectorXf predict_trajectory; //预测轨迹
    Eigen::VectorXf control_command;    //求解得到的控制量
    predict_trajectory.setZero(horizon * 12);
    control_command.setZero(horizon * 12);
    predict_trajectory = nmpc_controller.get_predict_trajectory();
    control_command = nmpc_controller.get_controls();

    // 更新状态和参考控制序列
    // 取下一次调用mpc时的状态和算出的这次的控制输入
    current_states = predict_trajectory.block(0 * 12, 0, 12, 1);
    desired_controls = control_command.block(0 * 12, 0, 12, 1);

    cout << "\n********* iteration_steps:" << i << " *********" << endl;
    cout << "current_states\n" << current_states.transpose() << endl;
    cout << "desired_states\n"
         << desired_states.block(0, 0, 12, 1).transpose() << endl;
    cout << "desired_controls\n"
         << desired_controls.transpose() << "\n\n"
         << endl;
  }
}
