// Copyright 2026 Dimensional Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <dimos_generated/messages.hpp>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>

int main(int argc, char** argv) {
    if (argc != 4) return 2;
    try {
        const std::string mode(argv[1]);
        dimos_msgs::msg::EpisodeStatus status{};
        if (mode == "seed" || mode == "seed-be") {
            status.ts = 17.25;
            status.state = "recording";
            status.episodes_saved = 2;
            status.episodes_discarded = 1;
            status.last_event = "start";
            status.task_label = {""};
        } else if (mode == "echo" || mode == "echo-be") {
            std::ifstream input(argv[2], std::ios::binary);
            if (!input) throw std::runtime_error("Cannot open input");
            const std::vector<uint8_t> bytes{std::istreambuf_iterator<char>(input), {}};
            status = dimos::cdr::decode<dimos_msgs::msg::EpisodeStatus>(bytes);
        } else {
            throw std::runtime_error("Expected seed, seed-be, echo or echo-be");
        }
        const auto bytes = dimos::cdr::encode(status, mode != "seed-be" && mode != "echo-be");
        std::ofstream output(argv[3], std::ios::binary);
        output.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
        if (!output) throw std::runtime_error("Cannot write output");
        std::cout << "C++ EpisodeStatus: " << status.state << ", labels=" << status.task_label.size() << '\n';
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
