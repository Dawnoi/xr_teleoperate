#include <cmath>
#include <iostream>
#include <sstream>
#include <string>

#include <unitree/robot/channel/channel_factory.hpp>
#include <unitree/robot/g1/agv/g1_agv_client.hpp>

namespace {

float clampf(float value, float lo, float hi) {
  return std::max(lo, std::min(hi, value));
}

}  // namespace

int main(int argc, char** argv) {
  const std::string network_interface = argc > 1 ? argv[1] : "";
  unitree::robot::ChannelFactory::Instance()->Init(0, network_interface);

  unitree::robot::g1::AgvClient client;
  client.SetTimeout(3.0F);
  client.Init();

  std::cout << "READY" << std::endl;
  std::cout.flush();

  std::string line;
  while (std::getline(std::cin, line)) {
    if (line.empty()) {
      continue;
    }

    std::istringstream iss(line);
    std::string cmd;
    iss >> cmd;

    if (cmd == "MOVE") {
      float vx = 0.0F, vy = 0.0F, vyaw = 0.0F;
      iss >> vx >> vy >> vyaw;
      vx = clampf(vx, -1.5F, 1.5F);
      vy = clampf(vy, -1.5F, 1.5F);
      vyaw = clampf(vyaw, -0.6F, 0.6F);
      const int32_t ret = client.Move(vx, vy, vyaw);
      std::cout << "OK MOVE " << ret << std::endl;
      std::cout.flush();
      continue;
    }

    if (cmd == "HEIGHT") {
      float vz = 0.0F;
      iss >> vz;
      vz = clampf(vz, -1.0F, 1.0F);
      const int32_t ret = client.HeightAdjust(vz);
      std::cout << "OK HEIGHT " << ret << std::endl;
      std::cout.flush();
      continue;
    }

    if (cmd == "STOP") {
      const int32_t ret_move = client.Move(0.0F, 0.0F, 0.0F);
      const int32_t ret_height = client.HeightAdjust(0.0F);
      std::cout << "OK STOP " << ret_move << " " << ret_height << std::endl;
      std::cout.flush();
      continue;
    }

    if (cmd == "PING") {
      std::cout << "PONG" << std::endl;
      std::cout.flush();
      continue;
    }

    if (cmd == "QUIT") {
      client.Move(0.0F, 0.0F, 0.0F);
      client.HeightAdjust(0.0F);
      std::cout << "BYE" << std::endl;
      std::cout.flush();
      break;
    }

    std::cout << "ERR unknown_command" << std::endl;
    std::cout.flush();
  }

  return 0;
}
