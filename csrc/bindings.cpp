#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include <vector>

#include "backends/ops.hpp"

namespace py = pybind11;

namespace {

torch::Tensor make_out_like(const torch::Tensor& in) {
    return torch::empty_like(in);
}

torch::Tensor make_equi_linear_out(const torch::Tensor& x, const torch::Tensor& weight) {
    if (x.sizes().size() < 2 || weight.sizes().size() < 2) {
        return torch::empty_like(x);
    }

    auto shape = x.sizes().vec();
    shape[shape.size() - 2] = weight.size(0);
    return torch::empty(shape, x.options());
}

}

PYBIND11_MODULE(turbogator_ext, m) {
    m.def("geometric_product_baseline", [](torch::Tensor a, torch::Tensor b) {
        if (a.numel() != b.numel()) {
            throw std::runtime_error("geometric_product_baseline: size mismatch");
        }
        auto out = make_out_like(a);
        tg::geometric_product_baseline(
            a.data_ptr<float>(),
            b.data_ptr<float>(),
            out.data_ptr<float>(),
            a.numel()
        );
        return out;
    });

    m.def("geometric_product_vectorized", [](torch::Tensor a, torch::Tensor b) {
        if (a.numel() != b.numel()) {
            throw std::runtime_error("geometric_product_vectorized: size mismatch");
        }
        auto out = make_out_like(a);
        tg::geometric_product_vectorized(
            a.data_ptr<float>(),
            b.data_ptr<float>(),
            out.data_ptr<float>(),
            a.numel()
        );
        return out;
    });

    m.def("equi_join_baseline", [](torch::Tensor a, torch::Tensor b, torch::Tensor ref) {
        if (a.numel() != b.numel()) {
            throw std::runtime_error("equi_join_baseline: size mismatch");
        }
        auto out = make_out_like(a);
        tg::equi_join_baseline(
            a.data_ptr<float>(),
            b.data_ptr<float>(),
            ref.numel() ? ref.data_ptr<float>() : nullptr,
            out.data_ptr<float>(),
            a.numel()
        );
        return out;
    });

    m.def(
        "equi_geometric_attention_baseline",
        [](torch::Tensor q, torch::Tensor k, torch::Tensor v, py::kwargs kwargs) {
            (void)kwargs;
            if (q.numel() != k.numel() || q.numel() != v.numel()) {
            throw std::runtime_error("equi_geometric_attention_baseline: size mismatch");
            }
            auto out = make_out_like(q);
            tg::equi_geometric_attention_baseline(
                q.data_ptr<float>(),
                k.data_ptr<float>(),
                v.data_ptr<float>(),
                out.data_ptr<float>(),
                q.numel()
            );
            return py::make_tuple(out, py::none());
        }
    );

    m.def(
        "scaler_gated_gelu_baseline",
        [](torch::Tensor x, py::object approximate) {
            (void)approximate;
        auto out = make_out_like(x);
        tg::scaler_gated_gelu_baseline(
            x.data_ptr<float>(),
            out.data_ptr<float>(),
            x.numel()
        );
        return out;
        },
        py::arg("x"),
        py::arg("approximate") = "tanh"
    );

    m.def(
        "equi_linear_baseline",
        [](torch::Tensor x, torch::Tensor weight, py::object bias, bool normalize_basis) {
            (void)normalize_basis;
            auto out = make_equi_linear_out(x, weight);
            const float* bias_ptr = nullptr;
            if (!bias.is_none()) {
                auto bias_tensor = bias.cast<torch::Tensor>();
                if (bias_tensor.numel()) {
                    bias_ptr = bias_tensor.data_ptr<float>();
                }
            }
            tg::equi_linear_baseline(
                x.data_ptr<float>(),
                weight.data_ptr<float>(),
                bias_ptr,
                out.data_ptr<float>(),
                out.numel()
            );
            return out;
        },
        py::arg("x"),
        py::arg("weight"),
        py::arg("bias") = py::none(),
        py::arg("normalize_basis") = true
    );

    m.def(
        "equi_rms_norm_baseline",
        [](torch::Tensor x, py::object weight, py::object eps) {
            (void)eps;
            auto out = make_out_like(x);
            const float* weight_ptr = nullptr;
            if (!weight.is_none()) {
                auto weight_tensor = weight.cast<torch::Tensor>();
                if (weight_tensor.numel()) {
                    weight_ptr = weight_tensor.data_ptr<float>();
                }
            }
            tg::equi_rms_norm_baseline(
                x.data_ptr<float>(),
                weight_ptr,
                out.data_ptr<float>(),
                out.numel()
            );
            return out;
        },
        py::arg("x"),
        py::arg("weight") = py::none(),
        py::arg("eps") = py::none()
    );
}
