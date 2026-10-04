#include <torch/extension.h>
#include <vector>
#include <tuple>
#include <string>
#include <cmath>

class EdgeGenerator {
public:
    int num_channels;
    float channel_radius;
    float time_radius;
    float time_window;
    int skip_channels;
    std::string features;

    EdgeGenerator(int num_channels, float channel_radius, float time_radius, 
                  float time_window, int skip_channels, std::string features)
        : num_channels(num_channels), channel_radius(channel_radius),
          time_radius(time_radius), time_window(time_window), 
          skip_channels(skip_channels), features(features) {}

    std::tuple<torch::Tensor, torch::Tensor> generate_edges(
        torch::Tensor times, torch::Tensor channels) 
    {
        std::vector<std::pair<int, int>> edges;
        std::vector<std::pair<float, float>> feature;
        std::vector<std::pair<float, int>> channel_last_event(num_channels, {0, -1});

        auto times_data = times.data_ptr<float>();
        auto channels_data = channels.data_ptr<float>();

        for (int idx = 0; idx < times.size(0); ++idx) {
            float time = times_data[idx*2];
            int channel = static_cast<int>(channels_data[idx*2]);

            float sum_t = 0;
            float sum_channel = 0;
            int sum_idx = 0;

            for (int n_channel = channel - static_cast<int>(channel_radius); 
                 n_channel <= channel + static_cast<int>(channel_radius); 
                 n_channel += skip_channels) 
            {
                if (n_channel < 0 || n_channel >= num_channels) {
                    continue;
                }

                if (channel_last_event[n_channel].second != -1) {
                    float n_time = channel_last_event[n_channel].first;
                    int n_idx = channel_last_event[n_channel].second;

                    if (time - n_time <= time_radius) {
                        edges.emplace_back(n_idx, idx);

                        if (features == "local") {
                            sum_t += (time - n_time);
                            sum_channel += (channel - n_channel);
                        } else if (features == "global") {
                            sum_t += n_time;
                            sum_channel += n_channel;
                        }

                        ++sum_idx;
                    }
                }
            }

            float mean_t = sum_idx > 0 ? std::round(sum_t / sum_idx) : 0;
            float mean_channel = sum_idx > 0 ? std::round(sum_channel / sum_idx) : 0;

            channel_last_event[channel] = {time, idx};

            if (features == "local") {
                feature.emplace_back(mean_t / time_radius, mean_channel / channel_radius);
            } else if (features == "global") {
                feature.emplace_back(mean_t / time_window, mean_channel / num_channels);
            }
        }

        torch::Tensor edge_tensor = torch::from_blob(edges.data(), {static_cast<long>(edges.size()), 2}, torch::kInt32).clone();
        torch::Tensor feature_tensor = torch::from_blob(feature.data(), {static_cast<float>(feature.size()), 2}, torch::kFloat32).clone();

        return {edge_tensor.t().contiguous(), feature_tensor};
    }
};

PYBIND11_MODULE(edge_generator, m) {
    pybind11::class_<EdgeGenerator>(m, "EdgeGenerator")
        .def(pybind11::init<int, float, float, float, int, std::string>())
        .def("generate_edges", &EdgeGenerator::generate_edges);
}
