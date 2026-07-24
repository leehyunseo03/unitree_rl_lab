#include "FSM/State_RLBase.h"
#include "unitree_articulation.h"
#include "isaaclab/envs/mdp/observations/observations.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <limits>
#include <unordered_map>

namespace
{

void print_controller_velocity_command(const isaaclab::ManagerBasedRLEnv* env)
{
    if (env == nullptr || env->robot->data.joystick == nullptr)
    {
        return;
    }

    auto & joystick = env->robot->data.joystick;
    const auto cfg = env->cfg["commands"]["base_velocity"]["ranges"];

    const std::array<float, 3> raw = {
        joystick->ly(),
        -joystick->lx(),
        -joystick->rx(),
    };
    const std::array<float, 3> cmd = {
        std::clamp(raw[0], cfg["lin_vel_x"][0].as<float>(), cfg["lin_vel_x"][1].as<float>()),
        std::clamp(raw[1], cfg["lin_vel_y"][0].as<float>(), cfg["lin_vel_y"][1].as<float>()),
        std::clamp(raw[2], cfg["ang_vel_z"][0].as<float>(), cfg["ang_vel_z"][1].as<float>()),
    };

    static std::array<float, 3> last_cmd = {
        std::numeric_limits<float>::quiet_NaN(),
        std::numeric_limits<float>::quiet_NaN(),
        std::numeric_limits<float>::quiet_NaN(),
    };
    static auto last_print = std::chrono::steady_clock::now() - std::chrono::seconds(1);

    const auto now = std::chrono::steady_clock::now();
    const bool changed = std::fabs(cmd[0] - last_cmd[0]) > 1e-3f ||
                         std::fabs(cmd[1] - last_cmd[1]) > 1e-3f ||
                         std::fabs(cmd[2] - last_cmd[2]) > 1e-3f;
    const bool periodic = now - last_print >= std::chrono::milliseconds(500);

    if (changed || periodic)
    {
        spdlog::info(
            "controller velocity command: lin_x={:.3f}, lin_y={:.3f}, yaw={:.3f} "
            "(raw ly={:.3f}, -lx={:.3f}, -rx={:.3f})",
            cmd[0], cmd[1], cmd[2], raw[0], raw[1], raw[2]);
        last_cmd = cmd;
        last_print = now;
    }
}

} // namespace

namespace isaaclab
{
// keyboard velocity commands example
// change "velocity_commands" observation name in policy deploy.yaml to "keyboard_velocity_commands"
REGISTER_OBSERVATION(keyboard_velocity_commands)
{
    std::string key = FSMState::keyboard->key();
    static auto cfg = env->cfg["commands"]["base_velocity"]["ranges"];

    static std::unordered_map<std::string, std::vector<float>> key_commands = {
        {"w", {1.0f, 0.0f, 0.0f}},
        {"s", {-1.0f, 0.0f, 0.0f}},
        {"a", {0.0f, 1.0f, 0.0f}},
        {"d", {0.0f, -1.0f, 0.0f}},
        {"q", {0.0f, 0.0f, 1.0f}},
        {"e", {0.0f, 0.0f, -1.0f}}
    };
    std::vector<float> cmd = {0.0f, 0.0f, 0.0f};
    if (key_commands.find(key) != key_commands.end())
    {
        // TODO: smooth and limit the velocity commands
        cmd = key_commands[key];
    }
    return cmd;
}

}

State_RLBase::State_RLBase(int state_mode, std::string state_string)
: FSMState(state_mode, state_string) 
{
    auto cfg = param::config["FSM"][state_string];
    auto policy_dir = param::parser_policy_dir(cfg["policy_dir"].as<std::string>());

    env = std::make_unique<isaaclab::ManagerBasedRLEnv>(
        YAML::LoadFile(policy_dir / "params" / "deploy.yaml"),
        std::make_shared<unitree::BaseArticulation<LowState_t::SharedPtr>>(FSMState::lowstate)
    );
    env->alg = std::make_unique<isaaclab::OrtRunner>(policy_dir / "exported" / "policy.onnx");

    this->registered_checks.emplace_back(
        std::make_pair(
            [&]()->bool{ return isaaclab::mdp::bad_orientation(env.get(), 1.0); },
            FSMStringMap.right.at("Passive")
        )
    );
}

void State_RLBase::run()
{
    print_controller_velocity_command(env.get());

    auto action = env->action_manager->processed_actions();
    for(int i(0); i < env->robot->data.joint_ids_map.size(); i++) {
        lowcmd->msg_.motor_cmd()[env->robot->data.joint_ids_map[i]].q() = action[i];
    }
}
