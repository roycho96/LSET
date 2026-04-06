// SPDX-License-Identifier: Apache-2.0
// PyTorch C++ extension bindings for SM100 fused contrastive loss kernel

#include <torch/extension.h>
#include <tuple>

// Forward declarations (defined in loss_sm100.cu)
std::tuple<torch::Tensor, torch::Tensor> fused_loss_fwd_sm100_impl(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor labels,
    float scale,
    int lse_mode
);

std::tuple<torch::Tensor, torch::Tensor> fused_loss_bwd_sm100_impl(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor labels,
    torch::Tensor ref_lse,
    torch::Tensor aux,
    torch::Tensor w,
    int loss_type
);

bool is_sm100_available();

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fwd", &fused_loss_fwd_sm100_impl,
          "Fused contrastive loss forward (SM100)",
          py::arg("Q"), py::arg("K"), py::arg("labels"),
          py::arg("scale"), py::arg("lse_mode"));

    m.def("bwd", &fused_loss_bwd_sm100_impl,
          "Fused contrastive loss backward (SM100)",
          py::arg("Q"), py::arg("K"), py::arg("labels"),
          py::arg("ref_lse"), py::arg("aux"), py::arg("w"),
          py::arg("loss_type"));

    m.def("is_sm100", &is_sm100_available,
          "Check if SM100+ GPU is available");
}
