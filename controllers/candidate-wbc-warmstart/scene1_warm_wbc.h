#pragma once

#include <humanoid_wbc/WeightedWbc.h>
#include <qpOASES.hpp>
#include <ros/ros.h>
#include <std_msgs/Float64MultiArray.h>
#include <array>
#include <algorithm>
#include <chrono>
#include <memory>
#include <stdexcept>

namespace ocs2 { namespace humanoid {

// Simulation-only selection is made explicitly by humanoidController.
// Existing task/constraint builders, settings and base-state updates are reused.
class Scene1WarmWbc final : public WeightedWbc {
 public:
  using WeightedWbc::WeightedWbc;

  void setEnabled(bool enabled) {
    if (enabled_ != enabled) solver_.reset();
    enabled_ = enabled;
  }

  vector_t update(const vector_t& state, const vector_t& input,
                  const vector_t& measured, size_t mode, scalar_t period,
                  bool mpc_update = false) override {
    if (!enabled_) {
      publish(false, 0., 0., 0., 0);
      return WeightedWbc::update(state, input, measured, mode, period, mpc_update);
    }
    WbcBase::update(state, input, measured, mode, period, mpc_update);
    const Task constraints = formulateConstraints(input);
    const Task task = formulateWeightedTasks(state, input, period);
    const int nv = static_cast<int>(getNumDecisionVars());
    const int neq = static_cast<int>(constraints.b_.size());
    const int nc = neq + static_cast<int>(constraints.f_.size());
    if (solver_ && (solver_->getNV() != nv || solver_->getNC() != nc || mode != last_mode_))
      solver_.reset();
    const int next = 1 - current_;
    Data& data = data_[next];
    data.a.resize(nc, nv);
    data.a << constraints.a_, constraints.d_;
    data.lower.resize(nc); data.upper.resize(nc);
    data.lower << constraints.b_, -qpOASES::INFTY * vector_t::Ones(nc - neq);
    data.upper << constraints.b_, constraints.f_;
    data.h = task.a_.transpose() * task.a_;
    data.g = -task.a_.transpose() * task.b_;
    if (!data.a.allFinite() || !data.h.allFinite() || !data.g.allFinite() ||
        !data.lower.allFinite() || !data.upper.allFinite())
      throw std::runtime_error("Non-finite WBC QP data");

    const auto started = std::chrono::steady_clock::now();
    bool hot = static_cast<bool>(solver_);
    int iterations = 200;
    qpOASES::real_t budget = .002;
    qpOASES::returnValue result = qpOASES::RET_QP_NOT_SOLVED;
    if (hot) {
      result = solver_->hotstart(data.h.data(), data.g.data(), data.a.data(),
          nullptr, nullptr, data.lower.data(), data.upper.data(), iterations, &budget);
      ++hot_solves_;
    }
    if (!hot || result != qpOASES::SUCCESSFUL_RETURN) {
      if (hot) ++fallbacks_;
      solver_.reset(new qpOASES::SQProblem(nv, nc));
      qpOASES::Options options;
      options.setToMPC();
      options.printLevel = qpOASES::PL_LOW;
      options.enableEqualities = qpOASES::BT_TRUE;
      solver_->setOptions(options);
      iterations = 200;
      // A cold initialization may use up to 10 ms to reach the same QP's
      // solution, instead of accepting an unfinished/stale output at 2 ms.
      budget = .010;
      result = solver_->init(data.h.data(), data.g.data(), data.a.data(),
          nullptr, nullptr, data.lower.data(), data.upper.data(), iterations, &budget);
      ++cold_solves_;
    }
    current_ = next;
    last_mode_ = mode;
    vector_t solution(nv);
    if (result != qpOASES::SUCCESSFUL_RETURN ||
        solver_->getPrimalSolution(solution.data()) != qpOASES::SUCCESSFUL_RETURN ||
        !solution.allFinite()) {
      ++invalid_outputs_;
      publish(hot, -1., -1., 0., iterations);
      throw std::runtime_error("WBC QP did not solve; no stale solution accepted");
    }
    const vector_t projected = data.a * solution;
    const double equality = neq ? (projected.head(neq) - constraints.b_).cwiseAbs().maxCoeff() : 0.;
    const double inequality = nc > neq ? std::max(0.,
        (projected.tail(nc - neq) - constraints.f_).maxCoeff()) : 0.;
    const double elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started).count();
    if (equality > 1e-5 || inequality > 1e-5) ++invalid_outputs_;
    publish(hot, equality, inequality, elapsed, iterations);
    if (invalid_outputs_)
      throw std::runtime_error("WBC QP constraint residual exceeded tolerance");
    return solution;
  }

 private:
  using RowMatrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  struct Data { RowMatrix h, a; vector_t g, lower, upper; };
  // qpOASES keeps shallow matrix references. Both old and new buffers must
  // remain alive throughout a matrix-changing hotstart, and after it returns.
  std::array<Data, 2> data_;
  std::unique_ptr<qpOASES::SQProblem> solver_;
  bool enabled_ = false;
  int current_ = 0;
  size_t last_mode_ = 0;
  unsigned long cold_solves_ = 0, hot_solves_ = 0, fallbacks_ = 0, invalid_outputs_ = 0;
  ros::Publisher trace_publisher_ = ros::NodeHandle().advertise<std_msgs::Float64MultiArray>(
      "/scene1_tracking/wbc_warmstart", 1);
  void publish(bool hot, double equality, double inequality, double elapsed, int iterations) {
    std_msgs::Float64MultiArray trace;
    trace.data = {(enabled_ ? 1. : 0.), (hot ? 1. : 0.), double(cold_solves_),
        double(hot_solves_), double(fallbacks_), double(invalid_outputs_),
        equality, inequality, elapsed, double(iterations)};
    trace_publisher_.publish(trace);
  }
};

}}  // namespace ocs2::humanoid
